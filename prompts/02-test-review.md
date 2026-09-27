# Test review prompt

Audit the test suite against `TASK.md`.

Identify requirements that are untested or only tested through mocks. Add the smallest meaningful CPU tests needed to cover high-risk behavior. Do not add tests that merely duplicate implementation details.

Verify that tests do not download models or datasets and can run without CUDA. Run the final test suite and report the exact command and result.
