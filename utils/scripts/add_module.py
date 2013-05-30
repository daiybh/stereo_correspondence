#!/usr/bin/python
# coding: utf-8

import os
import os.path
import sys
import datetime
import string


def _check2(txt,ret):
    print ('%s: ... [%s]' % (txt, 'OK' if ret else 'FAILED'))
    if not ret:
        raise RuntimeError('Failed')           
def _check(txt,ret):
    _check2('Checking %s'%txt,ret)
    
def _usage():
    print ('Usage: add_module.py module_name ClassName [namespace]')
    print ('')
    print ('\t module_name - name of the module (usually lower case with uderscores as a word separator')
    print ('\t ClassName - name of the class representing the module (usually CamelCase')
    print ('\t namespace - namespace to put he module to. if not provided, module_name will be used.')
    print ('')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        _usage();
        sys.exit(1)
    module = sys.argv[1]
    class_name = sys.argv[2]
    namespace = module
    if len(sys.argv) > 3:
        namespace = sys.argv[3]
    base = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_dir = os.path.join(base,'data')
    module_dir = os.path.abspath(os.path.join(base,'..','..','src','modules'))
    _check('module directory (%s) exists'%module_dir,os.path.exists(module_dir))
    target_dir = os.path.join(module_dir,module)
    _check('target directory  (%s) does not exist'%target_dir,lambda: not os.path.exists(target_dir))
    source = os.path.join(data_dir,'Module.cpp.tpl')
    header = os.path.join(data_dir,'Module.h.tpl')
    cmake = os.path.join(data_dir,'CMakeLists.txt.tpl')
    t_source = os.path.join(target_dir,'%s.cpp'%class_name)
    t_header = os.path.join(target_dir,'%s.h'%class_name)
    t_cmake = os.path.join(target_dir,'CMakeLists.txt')
    for x in [source, header, cmake]:
        _check('source template (%s) exists'%x,os.path.exists(x))
    
    print ('')
    print ('Creating module %s represented by class %s in namespace yuri::%s' % (module, class_name, namespace))
    date_today = datetime.date.today().strftime('%d.%m.%Y')
    guard = class_name.upper()
    mapping = {'module':module,
               'class_name':class_name,
               'namespace': namespace,
               'guard':guard,
               'date':date_today
               }
    
    module_h =  string.Template(open(header).read()).substitute(mapping)
    module_cpp =  string.Template(open(source).read()).substitute(mapping)
    cmake_txt = string.Template(open(cmake).read()).substitute(mapping)
    #_check2('Creating directory for new module',os.makedirs(target_dir))
    os.makedirs(target_dir)
    #_check2('Writing source',open(t_source,'w').write(module_cpp))                      
    #_check2('Writing header',open(t_header,'w').write(module_h))
    #_check2('Writing CMakeLists.txt',open(t_cmake,'w').write(cmake_txt))
    open(t_source,'w').write(module_cpp)                      
    open(t_header,'w').write(module_h)
    open(t_cmake,'w').write(cmake_txt)
    
