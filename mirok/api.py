import logging
import re
import traceback
from enum import Enum, unique

from codetoolkit.code_pos import CodePOS
from codetoolkit.delimiter import Delimiter

CODE_POS = CodePOS.get_inst()


@unique
class TokenType(Enum):
    OBJECT = "OBJECT"
    METHOD = "METHOD"
    OBJECT_TYPE = "OBJECT_TYPE"
    PARAM = "PARAM"
    PARAM_TYPE = "PARAM_TYPE"

@unique
class Tag(Enum):
    RES = "resource"
    OP1 = "operation 1"
    OP2 = "operation 2"
    NONE = "none"

class API:
    def __init__(self, object, method, args, ret):
        self.api_name = object[0] + '.' + method
        self.object = object # (type, name)
        self.method = method
        self.args = args  # (type, name)
        self.ret = ret
        self.token_list, self.token_type_list = [], []
        self.pos_list = []
        self.initialize()
        
    def __str__(self):
        return f"<API: {self.object} | {self.method} | {self.args} | {self.ret}>"

    def initialize(self):
        self.token_list = []
        self.token_type_list = []
        self.pos_list = []
        if self.object is not None:
            object = self.ret[1] if self.method == "<init>" and self.ret[1] != "<none>" else self.object[1]
            object = Delimiter.split_camel(object).strip()
        if self.method is not None:
            method = Delimiter.split_camel(self.method).strip() if self.method != "<init>" else "<init>"
        if len(object) != 0:
            for _ in object.split(" "):
                self.token_list.append(_)
                self.token_type_list.append(TokenType.OBJECT)
                self.pos_list.append('noun')
        if len(method) != 0:
            pos_tags = CODE_POS.tag(method) if method != "<init>" else [("<init>", "verb")]
            for token, pos in pos_tags:
                self.token_list.append(token)
                self.token_type_list.append(TokenType.METHOD)
                self.pos_list.append(pos)


    def match(self, res, op, relax=False):
        compound = " ".join(f"{token}-{type.value}-{pos}" for token, type, pos in zip(self.token_list, self.token_type_list, self.pos_list))
        '''resource in object part and operation in method part'''
        pattern = r"(^|.+ )(%s)( .+|) (%s)( .+|$)" % (
            " ".join(f"{re.escape(word)}-{TokenType.OBJECT.value}-noun" for word in res.split()),
            " ".join(f"{re.escape(word)}-{TokenType.METHOD.value}-verb" for word in op.split()))
        mobj = re.match(pattern, compound)
        
        if mobj:
            part1, part_res, part2, part_op, part3 = mobj.group(1), mobj.group(2), mobj.group(3), mobj.group(4), mobj.group(5)
            if len(part3) > 0 and part3.split()[0].endswith("-noun"):
                return None, None, (-1, -1), (-1, -1)

            res_start = part1.count("-") // 2
            res_end = res_start + part_res.count("-") // 2
            op_start = res_end + part2.count("-") // 2
            op_end = op_start + part_op.count("-") // 2
            cand_res = " ".join(s.split("-")[0] for s in part1.split() + part_res.split())
            pattern_obj = re.sub(r"\s+", " ", f"{part1} {part_res} {part2} {part3} VB-{TokenType.METHOD.value}-verb").strip() if not relax else \
                    re.sub(r"\s+", " ", f"{part1} {part_res} {part2} VB-{TokenType.METHOD.value}-verb").strip()
            # print(f'this {op_str}::{pattern}--{cand_res}::{"".join(self.token_list[op_start:op_end])}::::{"".join(self.token_list[res_start:res_end])}')
            return pattern_obj, cand_res, (res_start, res_end), (op_start, op_end)

        '''both operation and resource in method part'''
        pattern = r"(^|.+ )(%s)( .+|) (%s)( .+|$)" % (
            " ".join(f"{re.escape(word)}-{TokenType.METHOD.value}-verb" for word in op.split()),
            " ".join(f"{re.escape(word)}-{TokenType.METHOD.value}-noun" for word in res.split()))
        mobj = re.match(pattern, compound)
        if mobj:
            part1, part_op, part2, part_res, part3 = mobj.group(1), mobj.group(2), mobj.group(3), mobj.group(4), mobj.group(5)
            op_start = part1.count("-") // 2
            op_end = op_start + part_op.count("-") // 2
            res_start = op_end + part2.count("-") // 2
            res_end = res_start + part_res.count("-") // 2
            res_strs = part_res.split()
            for s in reversed(part2.split()):
                if s.endswith("-noun"):
                    res_strs.insert(0, s)
            for s in part2.split():
                if s.endswith("-noun"):
                    res_strs.append(s)
            cand_res = " ".join(s.split("-")[0] for s in res_strs)
            pattern_obj = re.sub(r"\s+", " ", f"{part1} VB-{TokenType.METHOD.value}-verb {part2} {part_res} {part3}").strip() if not relax else\
                    re.sub(r"\s+", " ", f"{part1} VB-{TokenType.METHOD.value}-verb {part2} {part_res}").strip()
            return pattern_obj, cand_res, (res_start, res_end), (op_start, op_end)
        return None, None, (-1, -1), (-1, -1)


    def decode(self, tag_list):
        res, op1, op2 = None, None, None
        has_res = False
        has_op = False
        for i in range(0, len(tag_list)):
            tag = tag_list[i]
            if tag == Tag.RES and not has_res:
                index = i
                res = ""
                while index < len(tag_list) and tag_list[index] == Tag.RES :
                    res += " " + self.token_list[index]
                    index += 1
                res = res.strip()
                has_res = True
            elif tag == Tag.OP1:
                op1 = self.token_list[i]
                has_op = True
            elif tag == Tag.OP2:
                op2 = self.token_list[i]
                has_op = True
        if has_op:
            return res, op1, op2
        else:
            return None, None, None
        

    
        
    
