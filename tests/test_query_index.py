import unittest
import contextlib
from io import StringIO

class TestQueryIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.query_index = __import__("query-index")

    def setUp(self):
        self.search = self.query_index.Search()

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def capture_stdout(cls, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            return f.getvalue(), ret

    def verify_line_structure(self, line, msg=None):
        msg_suffix = "no line structure" + ('' if msg is None else ' : ' + msg)
        if line is None:
            self.fail("line is None => " + msg_suffix)
        if not line.endswith("\n"):
            self.fail("line doesn't end with newline => " + msg_suffix)

    def verify_singleline_structure(self, line, msg=None):
        msg_suffix = "no singleline structure" + ('' if msg is None else ' : ' + msg)
        self.verify_line_structure(line, msg_suffix)
        count = line.count("\n")
        if count != 1:
            self.fail(f"{count} newlines instead of 1 => " + msg_suffix)

    def verify_command_behaviour_setfield(self, command_name, field_name,
        value, expect_success, success_msg_prefix=None, fail_msg_prefix=None,
        expect_iterationDone=True, expect_singleline_structure=True,
        interpolate_value=True):
        search = self.search
        setattr(search, field_name, None)
        search.in_text = f"{command_name} {value}" if interpolate_value else command_name
        msg_suffix = f"failed at command {command_name!r} data point {value} expect {'success' if expect_success else 'fail'}"

        output, iterationDone = self.capture_stdout(search.do_command)
        if expect_iterationDone is not None:
            self.assertEqual(expect_iterationDone, iterationDone, msg=f"Command {'did' if iterationDone else 'didnt'} request 'no search'. => {msg_suffix}")

        if expect_singleline_structure is not None:
            if not expect_singleline_structure:
                raise NotImplementedError("expect_singleline_structure=False not implemented => " + msg_suffix)
            self.verify_singleline_structure(output, msg=f"Command didn't return exactly a single line of output: {output!r} => {msg_suffix}")

        if expect_success is None:
            return
        expect_msg_prefix = success_msg_prefix if expect_success else fail_msg_prefix
        if expect_msg_prefix is not None:
            if not output.startswith(expect_msg_prefix):
                self.assertEqual(expect_msg_prefix, output, msg="Output message didn't start with expected message prefix. => " + msg_suffix)

        stored_value = getattr(search, field_name)
        if expect_success:
            self.assertEqual(value, stored_value, msg="Command didn't update Search object to data point value. => " + msg_suffix)
        else:
            self.assertIsNone(stored_value, msg="Command updated Search object although it shouldn't have. => " + msg_suffix)

    def test_command_q(self):
        self.verify_command_behaviour_setfield("q", "running_cli", False, True,
            expect_singleline_structure=None, interpolate_value=False)

    def test_command_ft(self):
        values = [
            (1.0, True), (0.5, True), (0.0, True),
            (1.1, False), (-0.1, False),
        ]
        for v in values:
            self.verify_command_behaviour_setfield("ft", "face_threshold", v[0],  v[1],
                success_msg_prefix="Set face similarity threshold",
                fail_msg_prefix="Invalid face threshold")
        # Check "show value" as well.
        search = self.search
        last_value = [v for v in filter(lambda v: v[1], values)][-1]  # Last to-be-successfully-set value.
        search.face_threshold = last_value  # Oh well... Necessary as negative tests reset the field to None!
        for command in ["ft", "ft show"]:
            search.in_text = command
            output, iterationDone = self.capture_stdout(search.do_command)
            self.assertTrue(iterationDone, msg=f"Command {command!r} didn't request 'no search'.")
            self.verify_singleline_structure(output, msg=f"Command {command!r} doesn't have single-line output. (output was: {output!r})")
            self.assertTrue(f"threshold is {last_value}" in output, msg=f"Command {command!r} didn't show current value. (output was: {output!r})")

    def test_command_h(self):
        search = self.search
        search.in_text = "h"
        output, iterationDone = self.capture_stdout(search.do_command)
        self.assertTrue(iterationDone, msg="Command didn't request 'no search'.")
        self.assertTrue(output.count("\n") >= 3, msg="Command didn't give multi-line output message.")
        self.assertTrue("Enter a search query" in output, msg="Command missing basic help string.")
        self.assertTrue("h\t\tShow this help" in output, msg="Command missing self-referential help string.")

if __name__ == '__main__':
    unittest.main()
