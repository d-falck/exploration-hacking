import ray

# Test local
print("Testing local environment... ", end="")
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print("✓ PASS")
    local_works = True
except Exception as e:
    print(f"✗ FAIL: {e}")
    local_works = False

# Test Ray worker
print("Testing Ray worker... ", end="")
ray.init(ignore_reinit_error=True)

@ray.remote
def test_worker():
    import flash_attn
    from flash_attn import flash_attn_func
    return "works"

try:
    ray.get(test_worker.remote())
    print("✓ PASS")
    ray_works = True
except Exception as e:
    print(f"✗ FAIL: {e}")
    ray_works = False

# Summary
print(f"\nLocal: {'✓' if local_works else '✗'}")
print(f"Ray:   {'✓' if ray_works else '✗'}")

ray.shutdown()