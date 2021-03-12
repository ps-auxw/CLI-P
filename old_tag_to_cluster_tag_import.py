import database
import config

database.open_db()
config.open_db()

tags = config.list_tags(False)
for tag in tags:
    tag_num, tag_name = tag
    print(f"Importing {tag_name}")
    contents = config.get_tag_contents(tag_name, False)
    for face in contents:
        fix_idx, face_id = face[0][0], face[0][1]
        config.add_tag(tag_name, fix_idx, face_id, True)
print("Done")
