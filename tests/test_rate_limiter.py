"""Tests for rate limiter."""

from sentinel.gateway.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(requests_per_minute=100, burst_size=10)
        assert limiter.allow("client-1") is True
        assert limiter.current_usage == 1

    def test_blocks_over_burst(self):
        limiter = RateLimiter(requests_per_minute=1000, burst_size=3)
        for _ in range(3):
            assert limiter.allow("client-1") is True
        assert limiter.allow("client-1") is False

    def test_different_clients_independent(self):
        limiter = RateLimiter(requests_per_minute=1000, burst_size=2)
        assert limiter.allow("client-1") is True
        assert limiter.allow("client-1") is True
        assert limiter.allow("client-1") is False
        assert limiter.allow("client-2") is True

    def test_reset(self):
        limiter = RateLimiter(burst_size=1)
        limiter.allow("client-1")
        limiter.reset()
        assert limiter.current_usage == 0
        assert limiter.allow("client-1") is True
