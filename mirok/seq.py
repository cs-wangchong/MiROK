from typing import List
from mirok.api import API, Tag

class Seq:
    def __init__(self, origin, api_seq:List[API]):
        self.origin = origin
        self.api_seq = api_seq  
        self.generate_token_seq()
        self.tag_seqs = []

    def generate_token_seq(self):
        self.token_seq, self.token_type_seq = [],[]
        for api in self.api_seq:
            token_list, type_list = api.token_list, api.token_type_list
            if token_list is not None:
                self.token_seq += token_list
                self.token_type_seq += type_list

    def valid(self):
        return len(self.tag_seqs) > 0

    def contains(self, res, op1, op2):
        acquired = set()
        released = set()
        for api in self.api_seq:
            pattern_obj1, *_ = api.match(res, op1, relax=True)
            pattern_obj2, *_ = api.match(res, op2, relax=True)
            if pattern_obj1 is not None:
                acquired.add(pattern_obj1)
            if pattern_obj2 is not None and pattern_obj2 in acquired:
                released.add(pattern_obj2)
            
        if len(acquired) > 0 and len(acquired - released) == 0:
            return True
        return False

    def expand_res(self, res, op1, op2):
        pattern_obj1 = None
        pattern_obj2 = None
        cand_res1 = None
        cand_res2 = None

        for api in self.api_seq:
            pattern_obj, cand_res, _, _ = api.match(res=res, op=op1)
            if pattern_obj is not None:
                pattern_obj1 = pattern_obj
                cand_res1 = cand_res
                continue
            pattern_obj, cand_res, _, _ = api.match(res=res, op=op2)
            if pattern_obj is not None:
                pattern_obj2 = pattern_obj
                cand_res2 = cand_res
                continue
                
        if cand_res1 == cand_res2 and pattern_obj1 is not None and pattern_obj2 is not None and pattern_obj1 == pattern_obj2:
            return cand_res1
        return None

    
    def annotate(self, res, op1, op2, conf=1):
        tag_seq = []
        pattern_obj1 = None
        pattern_obj2 = None

        for api in self.api_seq:
            api_tag_seq = [Tag.NONE] * len(api.token_list)
            pattern_obj, _, (res_start, res_end), (op_start, op_end) = api.match(res=res, op=op1)
            if pattern_obj is not None:
                pattern_obj1 = pattern_obj
                api_tag_seq[res_start:res_end] = [Tag.RES] * (res_end - res_start)
                api_tag_seq[op_start:op_end] = [Tag.OP1] * (op_end - op_start)
                tag_seq.extend(api_tag_seq)
                continue
            pattern_obj, _, (res_start, res_end), (op_start, op_end) = api.match(res=res, op=op2)
            if pattern_obj1 is not None and pattern_obj is not None: # ensure op2 behind op1
                pattern_obj2 = pattern_obj
                api_tag_seq[res_start:res_end] = [Tag.RES] * (res_end - res_start)
                api_tag_seq[op_start:op_end] = [Tag.OP2] * (op_end - op_start)
                tag_seq.extend(api_tag_seq)
                continue
            tag_seq.extend(api_tag_seq)
                
        if pattern_obj1 is not None and pattern_obj2 is not None and pattern_obj1 == pattern_obj2:
            self.tag_seqs.append((tag_seq, conf))
        
    
    def get_abs_rars_with_conf(self):
        abs_rars = set()
        for tag_seq, conf in self.tag_seqs:
            res, op1, op2 = None, None, None
            for api in self.api_seq:
                _res, _op1, _op2 = api.decode(tag_seq[:len(api.token_list)])
                res = _res if _res else res
                op1 = _op1 if _op1  else op1
                op2 = _op2 if op1 and _op2 else op2
                tag_seq = tag_seq[len(api.token_list):]
            
            if res and op1 and op2:
                abs_rars.add((res, op1, op2, conf))
                break
        return abs_rars

    def get_abs_rars(self):
        return {(res, op1, op2) for res, op1, op2, _ in self.get_abs_rars_with_conf()}

    def clear_tag(self):
        self.tag_seqs = list()

    def __str__(self):
        # return f"<Seq: api_seq={str([str(api) for api in self.api_seq])}, tag_seq={str([str(tag) for tag in self.tag_seq])}>"
        return f"<Seq origin:{self.origin}|\n\ttokens: {self.token_seq}\n\ttags:{self.tag_seqs}\n\ttoken types :{self.token_type_seq}>"
