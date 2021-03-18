import unittest
import contextlib
from io import StringIO

class TestQueryIndex(unittest.TestCase):

    def setUp(self):
        self.query_index = __import__("query-index")
        self.search = self.query_index.Search()

    def tearDown(self):
        pass

    @classmethod
    def capture_stdout(cls, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            return f.getvalue(), ret

    @classmethod
    def verify_line_structure(cls, line):
        if line is None:
            return False
        if not line.endswith("\n"):
            return False
        return True

    @classmethod
    def verify_singleline_structure(cls, line):
        if not cls.verify_line_structure(line):
            return False
        if line.count("\n") != 1:
            return False
        return True

    def test_command_ft(self):
        search = self.search
        success_msg_prefix = "Set face similarity threshold"
        fail_msg_prefix = "Invalid face threshold"

        def verify(value, expect_success):
            search.face_threshold = None
            search.in_text = f"ft {value}"
            assert_msg_prefix = f"data point {value}, expecting {'success' if expect_success else 'fail'}"

            output, iterationDone = self.capture_stdout(search.do_command)
            self.assertTrue(iterationDone, msg=assert_msg_prefix + ": Command didn't request 'no search'.")

            self.assertTrue(self.verify_singleline_structure(output), msg=assert_msg_prefix + ": Command didn't return exactly a single line of output: {repr(output)}")

            expect_msg_prefix = success_msg_prefix if expect_success else fail_msg_prefix
            if not output.startswith(expect_msg_prefix):
                self.assertEqual(expect_msg_prefix, output, msg=assert_msg_prefix + ": Output message didn't start with expected message prefix.")

            if expect_success:
                self.assertEqual(value, search.face_threshold, msg=assert_msg_prefix + ": Command didn't update Search object to data point value.")
            else:
                self.assertIsNone(search.face_threshold, msg=assert_msg_prefix + ": Command updated Search object although it shouldn't have.")

        values = [
            (1.0, True), (0.5, True), (0.0, True),
            (1.1, False), (-0.1, False),
        ]
        for v in values:
            verify(*v)

if __name__ == '__main__':
    unittest.main()
