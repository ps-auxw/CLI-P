import unittest
import contextlib
from io import StringIO
import os.path

class TestQueryIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path_prefix = "tests"
        cls.query_index = __import__("query-index")
        cls.db = cls.query_index.database.get(path_prefix=cls.path_prefix, pack_type='<L')
        cls.cfg = cls.query_index.config.get(path_prefix=cls.path_prefix)

    @classmethod
    def createSearchInstance(cls):
        return cls.query_index.Search(path_prefix=cls.path_prefix, db=cls.db, cfg=cls.cfg)

    def setUp(self):
        self.search = self.createSearchInstance()

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
        """Test query-index.Search text command 'q' (quit)."""
        self.verify_command_behaviour_setfield("q", "running_cli", False, True,
            expect_singleline_structure=None, interpolate_value=False)

    def test_command_ft(self):
        """Test query-index.Search text command 'ft' (face similarity threshold)."""
        search = self.search
        prev_value = search.face_threshold
        values = [
            (1.0, True), (0.6, True), (0.5, True), (0.0, True),
            (1.1, False), (-0.1, False),
        ]
        for value, expect_success in values:
            with self.subTest(value=value, expect_success=expect_success):
                self.verify_command_behaviour_setfield("ft", "face_threshold", value,  expect_success,
                    success_msg_prefix="Set face similarity threshold",
                    fail_msg_prefix="Invalid face threshold")
                # Also test for persistence:
                search2 = self.createSearchInstance()
                self.assertEqual(value if expect_success else prev_value, search2.face_threshold, msg=f"Persistence check for value {value}, expect_success {expect_success} failed.")
                if expect_success:
                    prev_value = search.face_threshold
        # Check "show value" as well.
        last_value = [v for v in filter(lambda v: v[1], values)][-1]  # Last to-be-successfully-set value.
        search.face_threshold = last_value  # Oh well... Necessary as negative tests reset the field to None!
        for command in ["ft", "ft show"]:
            with self.subTest(command=command):
                search.in_text = command
                output, iterationDone = self.capture_stdout(search.do_command)
                self.assertTrue(iterationDone, msg=f"Command {command!r} didn't request 'no search'.")
                self.verify_singleline_structure(output, msg=f"Command {command!r} doesn't have single-line output. (output was: {output!r})")
                self.assertTrue(f"threshold is {last_value}" in output, msg=f"Command {command!r} didn't show current value. (output was: {output!r})")

    def test_command_h(self):
        """Test query-index.Search text command 'h' (help)."""
        search = self.search
        search.in_text = "h"
        output, iterationDone = self.capture_stdout(search.do_command)
        self.assertTrue(iterationDone, msg="Command didn't request 'no search'.")
        self.assertTrue(output.count("\n") >= 3, msg="Command didn't give multi-line output message.")
        self.assertTrue("Enter a search query" in output, msg="Command missing basic help string.")
        self.assertTrue("h\t\tShow this help" in output, msg="Command missing self-referential help string.")

    def test_text_search(self):
        """Test query-index.Search text search on sample images."""
        search = self.search
        text2image_dict = {
            "owl":
                "dennis-buchner-wfFC7y5HY44-unsplash.jpg",
            "cat":
                "uriel-soberanes-xadzcCQZ_Xc-unsplash.jpg",
            "rabbit":
                "satyabrata-sm-u_kMWN-BWyU-unsplash.jpg",
            "forest": [
                "andrew-neel-a_K7R1kugUE-unsplash.jpg",
                "luca-bravo-ESkw2ayO2As-unsplash.jpg",
                "matt-dodd-1bywoXeKbT4-unsplash.jpg",
                #"priscilla-du-preez-XY9tbPYhR34-unsplash.jpg",  # This is not recognized as forest, but rather is a landscape after all.
                "satyabrata-sm-u_kMWN-BWyU-unsplash.jpg",  # ..rabbit is in forest, m'kay?
                "dennis-buchner-wfFC7y5HY44-unsplash.jpg",  # owl, too
                ],
            "forest at night":
                "matt-dodd-1bywoXeKbT4-unsplash.jpg",
            "church":
                "paul-teysen-FFXTiBQz42o-unsplash.jpg",
            "indoors":
                "adam-winger--OzQ6lO0mMA-unsplash.jpg",
            "city at night":
                "mike-swigunski-VEFEYV4M6mw-unsplash.jpg",
            "crowd":
                "jason-ortego-GbZsvIIi4Xw-unsplash.jpg",
        }
        known_false_positives = {
            "forest": [
                "uriel-soberanes-xadzcCQZ_Xc-unsplash.jpg",  # cat seems to be in garden, not forest, but close enough!
            ],
        }
        for search_text in text2image_dict:
            image_name = text2image_dict[search_text]
            false_positives = [] if search_text not in known_false_positives else known_false_positives[search_text]
            with self.subTest(search_text=search_text, image_name=image_name):
                search.in_text = search_text
                _, iterationDone = self.capture_stdout(search.do_command)
                self.assertFalse(iterationDone, msg=f"Search.do_command() requested 'no search' for search_text={search_text!r}")
                self.capture_stdout(search.do_search)
                if type(image_name) is str:
                    found = False
                    for result in search.prepare_results():
                        self.assertTrue(result.tfn.endswith('/' + image_name), msg=f"Search for search_text={search_text!r} didn't give expected result {image_name!r} but {result.tfn!r}")
                        found = True
                        break
                    self.assertTrue(found, msg=f"Search for search_text={search_text!r} gave no result!")
                else:
                    image_names_left = set(image_name)
                    n = 0
                    for result in search.prepare_results():
                        n += 1
                        result_image_name = os.path.basename(result.tfn)
                        found = False
                        try:
                            image_names_left.remove(result_image_name)
                            found = True
                        except KeyError:
                            if result_image_name in false_positives:
                                found = True
                        self.assertTrue(found, msg=f"Search for search_text={search_text!r} didn't give any expected result (at result number {n}, with image_names_left={image_names_left!r}), but {result_image_name!r}")
                        if len(image_names_left) == 0:
                            break
                    self.assertEqual(len(image_names_left), 0, msg=f"Search for search_text={search_text!r} has no further result (after {n} results, with image_names_left={image_names_left!r}")
                # TODO: Also check similarity score?

if __name__ == '__main__':
    unittest.main()
