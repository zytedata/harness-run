from dev.openrouter_endpoints import _base_provider_slug


def test_base_provider_slug_accepts_provider_and_endpoint_slugs():
    assert _base_provider_slug("moonshotai") == "moonshotai"
    assert _base_provider_slug("moonshotai/mxfp4") == "moonshotai"
    assert _base_provider_slug("google-vertex/us-east5") == "google-vertex"
