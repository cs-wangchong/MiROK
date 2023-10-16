
from collections import defaultdict
from pathlib import Path
import logging
from typing import List

from codetoolkit.javalang.parse import parse
from codetoolkit.javalang.tree import *

from mirok.absrar_base import AbsRARBase
from mirok.api import API
from mirok.seq import Seq

class RARFinder:
    def __init__(self, abs_rar_base: AbsRARBase):
        self.abs_rar_base = abs_rar_base
        self.op1_dict, self.op2_dict = defaultdict(set), defaultdict(set)
        for res, op1, op2 in self.abs_rar_base:
            self.op1_dict[op1].add((res, op1, op2))
            self.op2_dict[op2].add((res, op1, op2))
        self.op1s, self.op2s = set(self.op1_dict.keys()), set(self.op2_dict.keys())


    def extract_methods(self, code) -> List[Seq]:
        try:
            ast = parse(code)
        except:
            # logging.error("syntax error, continue to process next code snippet!")
            # logging.error(f'code: \n{code}')
            return []
        package = ast.package.name if ast.package else ""
        seqs = []
        for _, type_declaration in ast.filter((ClassDeclaration, InterfaceDeclaration)):
            apis = []
            type_name = type_declaration.name
            for _, node in type_declaration.filter((ConstructorDeclaration, MethodDeclaration)):
                if isinstance(node, ConstructorDeclaration):
                    api = API((type_name, type_name), "<init>", [], (type_name, "<none>"))
                    api.api_name = f"{package}.{api.api_name}"
                    apis.append(api)
                elif isinstance(node, MethodDeclaration):
                    ret_type = node.return_type.name if node.return_type else "<unknown>"
                    api = API((type_name, type_name), node.name, [], (ret_type, "<none>"))
                    api.api_name = f"{package}.{api.api_name}"
                    apis.append(api)
            if len(apis) >= 2:
                seqs.append(Seq(code, apis))
        return seqs
    
    def pre_match(self, seq: Seq):
        tokens = set(seq.token_seq)
        op1s = tokens & self.op1s
        op2s = tokens & self.op2s
        if len(op1s) == 0 or len(op2s) == 0:
            return set()
        candidate_abs_rars = set()
        for op1 in op1s:
            for op2 in op2s:
                candidate_abs_rars |= self.op1_dict[op1] & self.op2_dict[op2]
        return candidate_abs_rars
    
    def match(self, seq: Seq, res, op1, op2):
        acqs = dict()
        rels = dict()
        for api in seq.api_seq:
            pattern_obj1, *_ = api.match(res, op1, relax=False)
            pattern_obj2, *_ = api.match(res, op2, relax=False)
            if pattern_obj1 is not None:
                acqs[pattern_obj1] = api
                continue
            if pattern_obj2 is not None:
                rels[pattern_obj2] = api
            
        rars = set()
        for pattern_obj, api in acqs.items():
            if pattern_obj not in rels:
                continue
            rars.add((api, rels[pattern_obj]))
        return rars
    
    def find(self, lib_list: List):
        all_rars = set()
        
        for lib_name, lib_path in lib_list:
            logging.info(f"---------- Library: {lib_name} ----------")
            for path in Path(lib_path).rglob("*.java"):
                with path.open("r") as f:
                    code = f.read()
                seqs = self.extract_methods(code)
                for seq in seqs:
                    candidate_abs_rars = self.pre_match(seq) # for speed-up
                    for res, op1, op2 in candidate_abs_rars:
                        rars = self.match(seq, res, op1, op2)
                        for api1, api2 in rars:
                            logging.info(f"API1: {api1.api_name}, API2: {api2.api_name}, Abs-RAR: <{res}, {op1}, {op2}>")
                            all_rars.add((api1.api_name, api2.api_name))
        return all_rars
                