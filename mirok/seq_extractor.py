#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from typing import List
import logging

from codetoolkit.javalang.parse import parse
from codetoolkit.javalang.tree import *
from mirok.api import API
from mirok.seq import Seq

class SeqExtractor:
    
    @staticmethod
    def extract_variable_declarations(method_declaration: MethodDeclaration):
        var_map = dict()
        for parameter in method_declaration.parameters:
            var_map[parameter.name] = parameter.type.name
        for _, declaration_node in method_declaration.filter(LocalVariableDeclaration):
            for declarator in declaration_node.declarators:
                var_map[declarator.name] = declaration_node.type.name if not hasattr(declaration_node.type, 'sub_type') or not declaration_node.type.sub_type else declaration_node.type.sub_type.name
        return var_map
    
    
    @staticmethod
    def extract_arguments(inovcation, variable_map):
        args = []
        for argument in inovcation.arguments:
            if isinstance(argument, MemberReference):
                arg = argument.member
                if arg in variable_map.keys():
                    args.append((variable_map[arg], arg))
                else:
                    args.append(("<unknown>", arg))
        return args
    
    @staticmethod
    def extaract_for_method(method_declaration: MethodDeclaration):
        api_seq = list()
        # save lines which have been extracted as declaration to avoid extracted repeatly in invocation 
        visited = set()

        var_map = SeqExtractor.extract_variable_declarations(method_declaration)
        for _, node in method_declaration.filter((MethodInvocation, ClassCreator, Assignment, LocalVariableDeclaration)):
            if isinstance(node, LocalVariableDeclaration):
                for declarator in node.declarators:
                    if isinstance(declarator.initializer, MethodInvocation):
                        obj_name = '<none>' if declarator.initializer.qualifier == "" or declarator.initializer.qualifier is None else declarator.initializer.qualifier
                        md_name = declarator.initializer.member
                    elif isinstance(declarator.initializer, ClassCreator):
                        obj_name = declarator.initializer.type.name if declarator.initializer.type.name!=None else '<none>'    
                        md_name = "<init>"
                    else:
                        continue
                    obj_type  = var_map.get(obj_name, "<unknown>")
                    obj = (obj_type, obj_name)
                    args = SeqExtractor.extract_arguments(declarator.initializer, var_map)
                    ret = (node.type.name, declarator.name)
                    api = API(obj, md_name, args, ret)
                    api_seq.append(api)
                    visited.add(declarator.initializer.begin_pos[0])
                       
            elif isinstance(node, Assignment):
                if isinstance(node.value, MethodInvocation):
                    obj_name = '<none>' if node.value.qualifier == "" or node.value.qualifier is None else node.value.qualifier
                    md_name = node.value.member
                elif isinstance(node.value, ClassCreator):
                    obj_name = node.value.type.name if node.value.type.name!=None else '<none>'    
                    md_name = "<init>"
                else:
                    continue
                obj_type  = var_map.get(obj_name, "<unknown>")
                obj = (obj_type, obj_name)
                args = SeqExtractor.extract_arguments(node.value, var_map)
                if not isinstance(node.expressionl, MemberReference):
                    ret = ("<unknown>", "<none>")
                else:
                    ret = (var_map.get(node.expressionl.member, "<unknown>"), node.expressionl.member)
                api = API(obj, md_name, args, ret)
                api_seq.append(api)
                visited.add(node.value.begin_pos[0])
            
            elif node.begin_pos[0] not in visited: 
                if isinstance(node, MethodInvocation):
                    obj_name = '<none>' if node.qualifier == "" or node.qualifier is None else node.qualifier
                    md_name = node.member
                    ret = ("<unknown>", "<none>")
                elif isinstance(node, ClassCreator):
                    obj_name = node.type.name if node.type.name else '<none>'    
                    md_name = "<init>"
                    ret = (node.type.name, "<none>")
                else:
                    continue
                obj_type  = var_map.get(obj_name, "<unknown>")
                obj = (obj_type, obj_name)
                args = SeqExtractor.extract_arguments(node, var_map)
                api = API(obj, md_name, args, ret)
                api_seq.append(api)
            
        # for api in apis:
        #     print(api)
        return method_declaration.name, api_seq
    

    @staticmethod
    def extract(code, back_depth = 3) -> List[Seq]: 
        try:
            ast = parse(code)
        except:
            logging.error("syntax error, continue to process next code snippet!")
            logging.error(f'code: \n{code}')
            return []

        seqs = []
        method_dict = dict()
        for _, md in ast.filter(MethodDeclaration):
            name, api_seq = SeqExtractor.extaract_for_method(md)
            method_dict[name] = api_seq
            
        SeqExtractor.expand_call(method_dict, back_depth)
        for api_seq in method_dict.values():
            seq = Seq(code, api_seq)
            seqs.append(seq)
        return seqs


    @staticmethod
    def expand_call(method_dict, back_depth=3):
        if back_depth == 0:
            return
        merged = set()
        for name, apis in method_dict.items():
            positions = []
            replacements = []
            for idx, api in enumerate(apis):
                if api.method != name and api.method in method_dict:
                    positions.append(idx)
                    replacements.append(method_dict[api.method])
                    merged.add(api.method)
            new_apis = []
            begin_pos = 0
            for pos, rep in zip(positions, replacements):
                new_apis.extend(apis[begin_pos:pos])
                new_apis.extend(rep)
                begin_pos = pos + 1
            new_apis.extend(apis[begin_pos:len(apis)])
            method_dict[name] = new_apis
        for name in merged:
            method_dict.pop(name)
        if len(merged) == 0:
            return
        SeqExtractor.expand_call(method_dict, back_depth-1)



if __name__ == '__main__':
    #   public static final String SATURDAY;
         
    #     string s = "m.txt";
    #     String SUNDAY = "SUNDAY";
    #     float PI = 34;
        # final double PI = 3.14;
    code = '''
     public class Foo {
        Foo(int a){
            this.a = a;
        }
        void g() {
            MD md;
            byte[] digest = md.digest();
           DigestInputStream dis = new DigestInputStream(is, md); 
         }
        }
    '''
    print(SeqExtractor().extract_method_class(code))