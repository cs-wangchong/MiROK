

from collections import defaultdict
import csv

class AbsRARBase:
    def __init__(self):
        self.abs_rars = set()
        self.op1s = set()
        self.op2s = set()
        self.op1_count = defaultdict(int)
        self.op2_count = defaultdict(int)
        self.detect_triples = defaultdict(set)

    def __iter__(self):
        return iter(list(self.abs_rars))

    def add_abs_rar(self, res, op1, op2):
        self.abs_rars.add((res, op1, op2))
        self.op1s.add(op1)
        self.op2s.add(op2)
        self.op1_count[op1] += 1
        self.op2_count[op2] += 1

    def remove_abs_rar(self, res, op1, op2):
        if (res, op1, op2) in self.abs_rars:
            self.abs_rars.remove((res, op1, op2))
        self.op1_count[op1] -= 1
        self.op2_count[op2] -= 1
        if self.op1_count[op1] == 0:
            self.op1s.remove(op1)
        if self.op2_count[op2] == 0:
            self.op2s.remove(op2)

    def is_conflict(self, res, op1, op2):
        return op1 in self.op2s or op2 in self.op1s

    @classmethod
    def from_csv(cls, csv_path):
        res_kb = cls()
        with open(csv_path, "r", newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:    
                res_kb.add_abs_rar(row['resource'].strip(), row['operator1'].strip(), row['operator2'].strip())
        return res_kb
    
    def to_csv(self, csv_path):
        with open(csv_path, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["resource", "operation1", "operation2"])
            writer.writeheader()
            writer.writerows([{"resource":res, "operation1":op1, "operation2":op2} for res, op1, op2 in sorted(self.abs_rars)])

    def __str__(self):
        return "=============================== Abs-RAR KB ==============================\n" + \
                f'number of Abs-RAR: {len(self.abs_rars)}\n' + \
               '\n'.join(str(triple) for triple in sorted(self.abs_rars))

