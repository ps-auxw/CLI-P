## CLI-P test suite

This documents the ways the CLI-P test suite can be run.
For everything else, see the [main README](../README.md)!

All these commands have to be run *from the project top-level directory*;
that is, the commands should **not** be run from the tests directory,
otherwise they won't work and/or won't find the code to test...

*   Simply run the whole test-suite:

    (Dots as output for each test run, summary at the end.)

        CLI-P$ python -m unittest discover tests

    That is, invoke the Python `unittest` module's `discover` sub-command
    on the `tests` directory.

*   Run the whole test-suite verbosely:

    (Test names/descriptions will be printed during run.
    Example output as of 2021-03-23/-24.)

        CLI-P$ python -m unittest discover -v tests
        Removing LMDB database at tests/vectors.lmdb ...
        Removing LMDB database at tests/config.lmdb ...
        test_dbs_gone (test_001_clean.CleanTestEnvironment) ... ok
        test_scan_samples (test_002_build_index.TestBuildIndex) ... CLIPing sample-images/downscale...
        ............
        Preparing indexes...
        Generating matrix...
        Training index (12, 512)...
        WARNING clustering 12 points to 1 centroids: please provide at least 39 training points
        Adding to index...
        Training faces index (4, 512)...
        WARNING clustering 4 points to 1 centroids: please provide at least 39 training points
        Adding to faces index...
        Saving index...
        Saving faces index...
        Indexed 12 images and 4 faces.
        Done!
        ok
        test_command_ft (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'ft' (face similarity threshold). ... ok
        test_command_h (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'h' (help). ... ok
        test_command_q (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'q' (quit). ... ok
        test_text_search (test_query_index.TestQueryIndex)
        Test query-index.Search text search on sample images. ... ok

        ----------------------------------------------------------------------
        Ran 6 tests in 25.265s

        OK

*   Only run selected sub-tests:

    (Will likely only work if the `build-index` test case
    has been run at least once successfully before.)

        CLI-P$ python -m unittest discover -s tests -p test_query_index.py
        ....
        ----------------------------------------------------------------------
        Ran 4 tests in 6.274s

        OK

    That is, invoke the Python `unittest` module's `discover` sub-command
    with start directory `tests`, pattern for test files `test_query_index.py`.

    Or verbosely:

        CLI-P$ python -m unittest discover -v -s tests -p test_query_index.py
        test_command_ft (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'ft' (face similarity threshold). ... ok
        test_command_h (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'h' (help). ... ok
        test_command_q (test_query_index.TestQueryIndex)
        Test query-index.Search text command 'q' (quit). ... ok
        test_text_search (test_query_index.TestQueryIndex)
        Test query-index.Search text search on sample images. ... ok

        ----------------------------------------------------------------------
        Ran 4 tests in 6.139s

        OK

