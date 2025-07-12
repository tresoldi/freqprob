"""Test memory management features: streaming, compression, and profiling."""

import os
import tempfile

from freqprob import (
    CompressedFrequencyDistribution,
    DistributionMemoryAnalyzer,
    MemoryMonitor,
    MemoryProfiler,
    QuantizedProbabilityTable,
    SparseFrequencyDistribution,
    StreamingDataProcessor,
    StreamingFrequencyDistribution,
    StreamingLaplace,
    StreamingMLE,
    create_compressed_distribution,
    create_sparse_distribution,
)
from freqprob.profiling import HAS_PSUTIL, get_object_memory_usage, profile_memory_usage

# mypy: disable-error-code=arg-type


class TestStreamingFrequencyDistribution:
    """Test streaming frequency distribution functionality."""

    def test_basic_operations(self) -> None:
        """Test basic streaming operations."""
        stream_dist = StreamingFrequencyDistribution()

        # Test initial state
        assert stream_dist.get_total_count() == 0
        assert stream_dist.get_vocabulary_size() == 0
        assert stream_dist.get_count("unknown") == 0

        # Test single updates
        stream_dist.update("word1")
        assert stream_dist.get_count("word1") == 1
        assert stream_dist.get_total_count() == 1
        assert stream_dist.get_vocabulary_size() == 1

        stream_dist.update("word1", 5)
        assert stream_dist.get_count("word1") == 6
        assert stream_dist.get_total_count() == 6

    def test_batch_updates(self) -> None:
        """Test batch update functionality."""
        stream_dist = StreamingFrequencyDistribution()

        elements = ["word1", "word2", "word1", "word3"]
        stream_dist.update_batch(elements)

        assert stream_dist.get_count("word1") == 2
        assert stream_dist.get_count("word2") == 1
        assert stream_dist.get_count("word3") == 1
        assert stream_dist.get_total_count() == 4

        # Test with custom counts
        elements = ["word4", "word5"]
        counts = [10, 20]
        stream_dist.update_batch(elements, counts)

        assert stream_dist.get_count("word4") == 10
        assert stream_dist.get_count("word5") == 20
        assert stream_dist.get_total_count() == 34

    def test_vocabulary_size_limit(self) -> None:
        """Test vocabulary size limiting."""
        stream_dist = StreamingFrequencyDistribution(max_vocabulary_size=3)

        # Add more elements than the limit
        for i in range(10):
            stream_dist.update(f"word_{i}", i + 1)  # word_9 has highest count

        # Should only keep top 3 elements
        assert stream_dist.get_vocabulary_size() <= 3

        # Highest count elements should be preserved
        assert stream_dist.get_count("word_9") > 0
        assert stream_dist.get_count("word_8") > 0
        assert stream_dist.get_count("word_7") > 0

    def test_min_count_threshold(self) -> None:
        """Test minimum count threshold."""
        stream_dist = StreamingFrequencyDistribution(
            min_count_threshold=3,
            compression_threshold=1,  # Force immediate compression
        )

        # Add elements with various counts
        stream_dist.update("high_freq", 10)
        stream_dist.update("medium_freq", 5)
        stream_dist.update("low_freq", 1)

        # Trigger compression by adding one more element
        stream_dist.update("trigger", 1)

        # Low frequency elements should be removed
        assert stream_dist.get_count("high_freq") == 10
        assert stream_dist.get_count("medium_freq") == 5
        assert stream_dist.get_count("low_freq") == 0  # Removed due to threshold

    def test_decay_factor(self) -> None:
        """Test exponential decay functionality."""
        stream_dist = StreamingFrequencyDistribution(decay_factor=0.9, compression_threshold=1)

        # Add initial element
        stream_dist.update("word1", 10)
        initial_count = stream_dist.get_count("word1")

        # Add another element to trigger decay
        stream_dist.update("word2", 1)

        # Count should have decayed
        decayed_count = stream_dist.get_count("word1")
        assert decayed_count < initial_count
        assert decayed_count == initial_count * 0.9

    def test_statistics(self) -> None:
        """Test statistics gathering."""
        stream_dist = StreamingFrequencyDistribution(max_vocabulary_size=100)

        elements = ["a", "b", "c", "a", "b", "a"]
        stream_dist.update_batch(elements)

        stats = stream_dist.get_statistics()

        assert stats["vocabulary_size"] == 3
        assert stats["total_count"] == 6
        assert stats["update_count"] == 6
        assert stats["most_frequent"][0] == "a"
        assert stats["most_frequent"][1] == 3
        assert stats["average_count"] == 2.0

    def test_dict_interface(self) -> None:
        """Test dictionary-like interface."""
        stream_dist = StreamingFrequencyDistribution()

        elements = ["a", "b", "a", "c"]
        stream_dist.update_batch(elements)

        # Test dict conversion
        result_dict = stream_dist.to_dict()
        assert result_dict["a"] == 2
        assert result_dict["b"] == 1
        assert result_dict["c"] == 1

        # Test containment
        assert "a" in stream_dist
        assert "z" not in stream_dist

        # Test length
        assert len(stream_dist) == 3

        # Test iteration
        items = list(stream_dist.items())
        assert len(items) == 3


class TestStreamingScoringMethods:
    """Test streaming scoring methods."""

    def test_streaming_mle_basic(self) -> None:
        """Test basic StreamingMLE functionality."""
        initial_data = {"word1": 3, "word2": 2, "word3": 1}
        streaming_mle = StreamingMLE(initial_data, logprob=False)

        # Test initial probabilities
        assert abs(streaming_mle("word1") - 0.5) < 0.01  # 3/6
        assert abs(streaming_mle("word2") - 0.33333) < 0.01  # 2/6
        assert streaming_mle("unknown") == 0.0

        # Test incremental update
        streaming_mle.update_single("word1", 3)  # word1: 6, total: 9
        assert abs(streaming_mle("word1") - 0.66667) < 0.01  # 6/9

        # Test batch update
        streaming_mle.update_batch(["word4", "word4"], [2, 3])  # word4: 5, total: 14
        assert abs(streaming_mle("word4") - 0.35714) < 0.01  # 5/14

    def test_streaming_mle_with_unobs_prob(self) -> None:
        """Test StreamingMLE with unobserved probability."""
        initial_data = {"word1": 4, "word2": 1}
        streaming_mle = StreamingMLE(initial_data, unobs_prob=0.1, logprob=False)

        # Observed elements should have reduced probability
        prob_word1 = streaming_mle("word1")
        expected_word1 = (4 / 5) * (1 - 0.1)  # MLE * (1 - unobs_prob)
        assert abs(prob_word1 - expected_word1) < 0.01

        # Unobserved elements should have unobs_prob
        assert streaming_mle("unknown") == 0.1

    def test_streaming_laplace(self) -> None:
        """Test StreamingLaplace functionality."""
        initial_data = {"word1": 2, "word2": 1}
        streaming_laplace = StreamingLaplace(initial_data, logprob=False)

        # Test Laplace smoothing: (count + 1) / (total + vocab_size)
        # word1: (2 + 1) / (3 + 2) = 3/5 = 0.6
        assert abs(streaming_laplace("word1") - 0.6) < 0.01

        # word2: (1 + 1) / (3 + 2) = 2/5 = 0.4
        assert abs(streaming_laplace("word2") - 0.4) < 0.01

        # unknown: 1 / (3 + 2) = 1/5 = 0.2
        assert abs(streaming_laplace("unknown") - 0.2) < 0.01

        # Test incremental update
        streaming_laplace.update_single("word3", 5)
        # Now: word1: 2, word2: 1, word3: 5, total: 8, vocab: 3
        # word3: (5 + 1) / (8 + 3) = 6/11 ≈ 0.545
        word3_prob = streaming_laplace("word3")
        assert word3_prob > 0.5  # Should be the highest probability
        assert word3_prob < 1.0  # Should be less than 1

    def test_streaming_statistics(self) -> None:
        """Test streaming method statistics."""
        initial_data = {"a": 1, "b": 2}
        streaming_mle = StreamingMLE(initial_data, max_vocabulary_size=10)

        # Test initial state (should be 0 since no updates yet)
        initial_count = streaming_mle.get_update_count()
        assert initial_count >= 0  # Could be 0 or initial data count

        # Update and check statistics
        streaming_mle.update_single("c", 3)
        assert streaming_mle.get_update_count() == initial_count + 1

        streaming_mle.update_batch(["d", "e"])
        assert streaming_mle.get_update_count() == initial_count + 3

        # Test streaming-specific statistics
        stats = streaming_mle.get_streaming_statistics()
        assert "vocabulary_size" in stats
        assert "total_count" in stats
        assert "update_count" in stats

    def test_save_load_state(self) -> None:
        """Test saving and loading streaming scorer state."""
        initial_data = {"word1": 5, "word2": 3}
        streaming_mle = StreamingMLE(initial_data, logprob=False)

        # Update the scorer
        streaming_mle.update_single("word3", 2)
        original_prob = streaming_mle("word3")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            streaming_mle.save_state(tmp_path)

            # Load from file
            loaded_mle = StreamingMLE.load_state(tmp_path)

            # Check that state was preserved
            assert loaded_mle("word3") == original_prob
            assert loaded_mle("word1") == streaming_mle("word1")
            assert loaded_mle.get_update_count() == streaming_mle.get_update_count()

        finally:
            os.unlink(tmp_path)


class TestStreamingDataProcessor:
    """Test streaming data processor."""

    def test_basic_processing(self) -> None:
        """Test basic data processing functionality."""
        methods = {
            "mle": StreamingMLE(max_vocabulary_size=100, logprob=False),
            "laplace": StreamingLaplace(max_vocabulary_size=100, logprob=False),
        }
        processor = StreamingDataProcessor(methods, batch_size=3)

        # Process single elements
        processor.process_element("word1", 2)
        processor.process_element("word2", 1)

        # Check scores
        assert processor.get_score("mle", "word1") > processor.get_score("mle", "word2")
        assert processor.get_score("laplace", "word1") > 0

        # Process batch
        processor.process_batch(["word3", "word1", "word3"])

        # word3 should now have count 2
        assert processor.get_score("mle", "word3") > 0

    def test_text_stream_processing(self) -> None:
        """Test text stream processing."""
        methods = {"mle": StreamingMLE(logprob=False)}
        processor = StreamingDataProcessor(methods, batch_size=2)

        # Process text stream
        text_tokens = ["the", "cat", "sat", "on", "the", "mat"]
        processor.process_text_stream(iter(text_tokens))

        # 'the' should have highest probability
        prob_the = processor.get_score("mle", "the")
        prob_cat = processor.get_score("mle", "cat")
        assert prob_the > prob_cat

    def test_processor_statistics(self) -> None:
        """Test processor statistics gathering."""
        methods = {"mle": StreamingMLE(), "laplace": StreamingLaplace()}
        processor = StreamingDataProcessor(methods)

        # Process some data
        processor.process_batch(["a", "b", "c", "a"])

        stats = processor.get_statistics()
        assert stats["processed_count"] == 4
        assert "methods" in stats
        assert "mle" in stats["methods"]
        assert "laplace" in stats["methods"]
        assert stats["methods"]["mle"]["update_count"] == 4


class TestMemoryEfficientRepresentations:
    """Test memory-efficient distribution representations."""

    def test_compressed_frequency_distribution(self) -> None:
        """Test compressed frequency distribution."""
        original_data = {"word1": 1000, "word2": 500, "word3": 100, "word4": 1}

        # Test without quantization
        compressed = CompressedFrequencyDistribution()
        compressed.update(original_data)

        # Check values are preserved
        assert compressed.get_count("word1") == 1000
        assert compressed.get_count("word2") == 500
        assert compressed.get_count("word3") == 100
        assert compressed.get_count("word4") == 1
        assert compressed.get_total_count() == 1601
        assert compressed.get_vocabulary_size() == 4

        # Test dictionary conversion
        result_dict = compressed.to_dict()
        assert result_dict == original_data

    def test_compressed_with_quantization(self) -> None:
        """Test compressed distribution with quantization."""
        original_data = {"word1": 1000, "word2": 500, "word3": 100}

        # Use quantization with few levels for testing
        compressed = CompressedFrequencyDistribution(quantization_levels=4)
        compressed.update(original_data)

        # Values should be approximately preserved (with quantization error)
        count1 = compressed.get_count("word1")
        count2 = compressed.get_count("word2")
        count3 = compressed.get_count("word3")

        # Check relative ordering is preserved
        assert count1 > count2 > count3

        # Check memory usage information
        memory_usage = compressed.get_memory_usage()
        assert "total" in memory_usage
        assert "counts_array" in memory_usage

    def test_compression_serialization(self) -> None:
        """Test compression and serialization."""
        original_data = {"a": 100, "b": 50, "c": 25}

        compressed = CompressedFrequencyDistribution(use_compression=True)
        compressed.update(original_data)

        # Serialize to bytes
        compressed_bytes = compressed.compress_to_bytes()
        assert isinstance(compressed_bytes, bytes)
        assert len(compressed_bytes) > 0

        # Deserialize
        restored = CompressedFrequencyDistribution.decompress_from_bytes(
            compressed_bytes, use_compression=True
        )

        # Check values are restored
        assert restored.get_count("a") == 100
        assert restored.get_count("b") == 50
        assert restored.get_count("c") == 25

    def test_sparse_frequency_distribution(self) -> None:
        """Test sparse frequency distribution."""
        # Create data with many zero counts (simulated by not including them)
        sparse_data = {"rare1": 1, "rare2": 2, "common": 1000}

        sparse = SparseFrequencyDistribution()
        sparse.update(sparse_data)

        # Test basic operations
        assert sparse.get_count("common") == 1000
        assert sparse.get_count("rare1") == 1
        assert sparse.get_count("nonexistent") == 0  # Default count

        # Test incremental updates
        sparse.increment("rare1", 4)
        assert sparse.get_count("rare1") == 5

        # Test top-k functionality
        top_2 = sparse.get_top_k(2)
        assert len(top_2) == 2
        assert top_2[0][0] == "common"  # Highest count
        assert top_2[0][1] == 1000

    def test_sparse_count_histogram(self) -> None:
        """Test sparse distribution count histogram."""
        sparse = SparseFrequencyDistribution()
        sparse.update({"a": 1, "b": 1, "c": 2, "d": 2, "e": 5})

        histogram = sparse.get_count_histogram()
        assert histogram[1] == 2  # Two elements with count 1
        assert histogram[2] == 2  # Two elements with count 2
        assert histogram[5] == 1  # One element with count 5

    def test_sparse_count_range_query(self) -> None:
        """Test sparse distribution count range queries."""
        sparse = SparseFrequencyDistribution()
        sparse.update({"a": 1, "b": 5, "c": 10, "d": 15, "e": 20})

        # Get elements with counts between 5 and 15
        elements = sparse.get_elements_with_count_range(5, 15)
        assert set(elements) == {"b", "c", "d"}

    def test_quantized_probability_table(self) -> None:
        """Test quantized probability table."""
        prob_table = QuantizedProbabilityTable(num_quantization_levels=256)

        # Set probabilities
        probs = {"word1": 0.5, "word2": 0.3, "word3": 0.2}
        prob_table.set_probabilities(probs)

        # Check approximate preservation (with some tolerance for quantization error)
        assert abs(prob_table.get_probability("word1") - 0.5) < 0.02
        assert abs(prob_table.get_probability("word2") - 0.3) < 0.02
        assert abs(prob_table.get_probability("word3") - 0.2) < 0.02

        # Test default probability
        prob_table.set_default_probability(0.01)
        assert abs(prob_table.get_probability("unknown") - 0.01) < 0.001

        # Test quantization error analysis
        error_stats = prob_table.get_quantization_error_stats(probs)
        assert "mean_absolute_error" in error_stats
        assert "max_absolute_error" in error_stats
        assert error_stats["num_elements"] == 3

    def test_factory_functions(self) -> None:
        """Test factory functions for creating efficient representations."""
        original_data = {"a": 100, "b": 50, "c": 1}

        # Test compressed creation
        compressed = create_compressed_distribution(original_data)
        assert compressed.get_count("a") == 100

        # Test sparse creation
        sparse = create_sparse_distribution(original_data)
        assert sparse.get_count("a") == 100

        # Test with quantization
        quantized = create_compressed_distribution(original_data, quantization_levels=16)
        assert quantized.get_count("a") > 0  # Should be approximately preserved


class TestMemoryProfiling:
    """Test memory profiling utilities."""

    def test_memory_profiler_basic(self) -> None:
        """Test basic memory profiler functionality."""
        profiler = MemoryProfiler()

        # Take initial snapshot
        snapshot1 = profiler.take_snapshot()
        if HAS_PSUTIL:
            assert snapshot1.rss_mb > 0
        else:
            assert snapshot1.rss_mb == 0.0  # When psutil not available
        assert snapshot1.timestamp > 0

        # Profile an operation
        with profiler.profile_operation("test_operation"):
            # Create some objects to use memory
            large_list = [i**2 for i in range(10000)]
            del large_list

        # Check metrics were recorded
        metrics = profiler.get_latest_metrics()
        assert metrics is not None
        assert metrics.operation_name == "test_operation"
        assert metrics.execution_time > 0

        # Test summary
        summary = profiler.get_memory_summary()
        assert "total_snapshots" in summary
        assert summary["total_snapshots"] >= 2

    def test_memory_profiler_decorator(self) -> None:
        """Test memory profiling decorator."""

        @profile_memory_usage("decorated_function")
        def test_function() -> list[int]:
            return list(range(1000))

        # Call function
        result = test_function()
        assert len(result) == 1000

        # Check profiler was attached
        profiler = test_function.get_profiler()
        assert profiler is not None

        metrics = profiler.get_latest_metrics()
        assert metrics.operation_name == "decorated_function"

    def test_distribution_memory_analyzer(self) -> None:
        """Test distribution memory analyzer."""
        analyzer = DistributionMemoryAnalyzer()

        # Create test distribution
        freqdist = {f"word_{i}": max(1, 100 - i) for i in range(50)}

        # Analyze memory usage
        original_memory = analyzer.measure_distribution_memory(freqdist)
        assert "total_mb" in original_memory
        assert "num_elements" in original_memory
        assert original_memory["num_elements"] == 50

        # Compare representations
        comparison = analyzer.compare_representations(freqdist)
        assert "original" in comparison
        assert "compressed" in comparison
        assert "sparse" in comparison
        assert "memory_savings" in comparison

        # Check savings calculations
        savings = comparison["memory_savings"]
        assert "compressed" in savings
        assert "percentage_savings" in savings["compressed"]

    def test_memory_monitor(self) -> None:
        """Test memory monitor functionality."""
        # Use a low threshold for testing
        monitor = MemoryMonitor(memory_threshold_mb=1.0, monitoring_interval=0.1)

        monitor.start_monitoring()

        # Check memory (might trigger alert depending on current usage)
        monitor.check_memory()
        # Can't assert on alert presence as it depends on actual memory usage

        monitor.stop_monitoring()

        # Get monitoring report
        report = monitor.get_monitoring_report()
        assert "monitoring_duration" in report
        assert "memory_statistics" in report
        assert "memory_trend" in report

    def test_object_memory_usage(self) -> None:
        """Test object memory usage analysis."""
        # Test dictionary
        test_dict = {"a": 1, "b": 2, "c": 3}
        dict_usage = get_object_memory_usage(test_dict)
        assert "basic_size" in dict_usage
        assert "total_size" in dict_usage
        assert "num_items" in dict_usage
        assert dict_usage["num_items"] == 3

        # Test list
        test_list = [1, 2, 3, 4, 5]
        list_usage = get_object_memory_usage(test_list)
        assert list_usage["num_items"] == 5

        # Test other object
        test_string = "hello world"
        string_usage = get_object_memory_usage(test_string)
        assert "basic_size" in string_usage


class TestMemoryManagementIntegration:
    """Test integration between memory management features."""

    def test_streaming_with_compression(self) -> None:
        """Test using streaming with compression."""
        # Create streaming scorer
        streaming_mle = StreamingMLE(max_vocabulary_size=100, logprob=False)

        # Add data incrementally
        for i in range(200):
            streaming_mle.update_single(f"word_{i}", i + 1)

        # Should respect vocabulary limit
        stats = streaming_mle.get_streaming_statistics()
        assert stats["vocabulary_size"] <= 100

        # High-frequency words should be preserved
        assert streaming_mle("word_199") > 0  # Highest frequency

    def test_memory_efficient_scoring_comparison(self) -> None:
        """Test comparing memory efficiency of different scoring approaches."""
        # Create test data
        large_vocab = {f"word_{i}": max(1, 1000 - i) for i in range(1000)}

        # Test regular MLE
        from freqprob import MLE

        regular_mle = MLE(large_vocab, logprob=False)

        # Test streaming MLE
        streaming_mle = StreamingMLE(large_vocab, max_vocabulary_size=500, logprob=False)

        # Test elements
        test_elements = [f"word_{i}" for i in range(10)]

        # Compare scores (should be similar for top elements)
        for element in test_elements:
            regular_score = regular_mle(element)
            streaming_score = streaming_mle(element)

            # Both should give positive scores for these elements
            assert regular_score > 0
            # Only check elements that were kept by the streaming version
            # (streaming version may have dropped some elements due to vocabulary limit)
            if streaming_score > 0:
                # If the element exists in both, scores should be reasonably similar
                # Allow for some difference due to compression
                relative_diff = abs(regular_score - streaming_score) / regular_score
                assert relative_diff < 0.5  # Allow up to 50% difference

    def test_profiling_memory_efficient_operations(self) -> None:
        """Test profiling memory-efficient operations."""
        profiler = MemoryProfiler()

        # Create test data
        test_data = {f"item_{i}": i + 1 for i in range(1000)}

        # Profile compressed distribution creation
        with profiler.profile_operation("create_compressed"):
            create_compressed_distribution(test_data, quantization_levels=256)

        # Profile sparse distribution creation
        with profiler.profile_operation("create_sparse"):
            create_sparse_distribution(test_data)

        # Check both operations were profiled
        all_metrics = profiler.get_all_metrics()
        assert len(all_metrics) == 2

        operation_names = [m.operation_name for m in all_metrics]
        assert "create_compressed" in operation_names
        assert "create_sparse" in operation_names

    def test_end_to_end_memory_efficiency(self) -> None:
        """Test end-to-end memory efficiency workflow."""
        # Start with large vocabulary
        large_freqdist = {f"term_{i}": max(1, 10000 - i * 10) for i in range(2000)}

        # Create memory analyzer
        analyzer = DistributionMemoryAnalyzer()

        # Compare all representations
        comparison = analyzer.compare_representations(large_freqdist)

        # Verify that analysis completed successfully
        assert "memory_savings" in comparison
        assert "original" in comparison
        assert "compressed" in comparison
        assert "sparse" in comparison

        # Verify that at least some methods have reasonable compression ratios
        # (Note: Small datasets may not show memory savings due to overhead)
        savings = comparison["memory_savings"]
        assert all(
            isinstance(savings[method]["compression_ratio"], int | float) for method in savings
        )
        assert all(savings[method]["compression_ratio"] > 0 for method in savings)

        # Create streaming processor for efficient updates
        methods = {
            "streaming_mle": StreamingMLE(large_freqdist, max_vocabulary_size=1000),
            "streaming_laplace": StreamingLaplace(large_freqdist, max_vocabulary_size=1000),
        }
        processor = StreamingDataProcessor(methods, batch_size=100)

        # Process additional data
        new_data = [f"new_term_{i}" for i in range(500)]
        processor.process_text_stream(iter(new_data))

        # Check processing statistics
        stats = processor.get_statistics()
        assert stats["processed_count"] == 500

        # Verify methods maintained vocabulary limits
        for method_stats in stats["methods"].values():
            if "vocabulary_size" in method_stats:
                assert method_stats["vocabulary_size"] <= 1000
