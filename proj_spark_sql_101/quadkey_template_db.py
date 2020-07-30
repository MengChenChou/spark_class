class Node:
    def __init__(self, digit):
        self.digit = digit
        self.region_names = []
        self.links = [None, None, None, None]

    def add_region(self, region_name):
        self.region_names.append(region_name)

    def add_node(self, digit):
        child = Node(digit)
        self.links[digit] = child
        return child


class QuadkeyTemplateDB:
    def __init__(self, csv_file):
        self.root = Node(-1)
        self.load_data(csv_file)

    def load_data(self, csv_file):
        file_handle = open(csv_file)
        for line in file_handle:
            row = line.strip().split(",")
            region_name = row[0]
            quadkey = row[2]
            self.add_quadkey(quadkey, region_name)

    def add_quadkey(self, quadkey, region_name):
        cur = self.root

        for i in quadkey:
            digit = int(i)
            if cur.links[digit] is None:
                cur = cur.add_node(digit)
            else:
                cur = cur.links[digit]
        cur.add_region(region_name)

    def lookup_regions(self, quadkey):
        if len(quadkey) != 15 or quadkey is None:
            return []
        cur = self.root
        region_found = []
        for i in quadkey:
            digit = int(i)
            if cur.links[digit] is not None:
                cur = cur.links[digit]
                if len(cur.region_names) > 0:
                    region_found += (cur.region_names)
            else:
                break
        return region_found


if __name__ == "__main__":
    db = QuadkeyTemplateDB("/home/cloudera/PythonProjects/proj_audi_case_study/region_template/qk_cn.csv")
    print(db.lookup_regions("132100103322203"))