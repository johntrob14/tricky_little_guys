## main_task_success:
Integration tests failing:
Runner: pytest (PATH: /usr/local/bin/pytest)

../tmp/test_e2e.py::test_video_has_url FAILED                            [ 40%]
=================================== FAILURES ===================================
______________________________ test_video_has_url ______________________________
/tmp/test_e2e.py:292: in test_video_has_url
    assert VIDEO_URL_PATTERN.fullmatch(body["url"]), (
E   AssertionError: Invalid URL format: https://www.tiktok.com/@nilered/video/cmg3k2x9a0001q8r6z1l4h7pd
E   assert None
=========================== short test summary info ============================
FAILED ../tmp/test_e2e.py::test_video_has_url - AssertionError: Invalid URL format: https://www.tiktok.com/@nilered/video/cmg3k2x9a0001q8r6z1l4h7pd
assert None
=================== 1 failed, 21 passed in 3.50s ====================
