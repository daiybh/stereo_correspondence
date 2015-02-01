#!/usr/bin/python
# coding: utf-8

import os
import os.path
import sys
import datetime
import string
import subprocess


def _check2(txt,ret):
    print ('%s: ... [%s]' % (txt, 'OK' if ret else 'FAILED'))
    if not ret:
        raise RuntimeError('Failed')           
def _check(txt,ret):
    _check2('Checking %s'%txt,ret)
    
def _get_git_value(value, default_value):
    try:
        v = subprocess.check_output(['/usr/bin/git','config','--get',value])
        v = v.decode().strip()
        if len(v) > 0:
            return v
    except:
        pass
    return default_value

def _usage():
    print ('Usage: add_module.py [--type=type] module_name ClassName [namespace]')
    print ('')
    print ('\t type - type of module (use --type=help to list supported types)')
    print ('\t module_name - name of the module (usually lower case with uderscores as a word separator')
    print ('\t ClassName - name of the class representing the module (usually CamelCase')
    print ('\t namespace - namespace to put he module to. if not provided, module_name will be used.')
    print ('')

types = {'thread': 
            {'description': 'Generic thread, for input nodes and event only nodes',
             'cmake': 'CMakeLists.txt.tpl', 'source':'Thread.cpp.tpl', 'header':'Thread.h.tpl'},
         'filter': 
            {'description': 'Filter thread, for processing and output nodes.',
             'cmake': 'CMakeLists.txt.tpl', 'source':'Filter.cpp.tpl', 'header':'Filter.h.tpl'}
         }

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        _usage();
        sys.exit(1)
    type = 'thread'
    if sys.argv[1][0:7] == '--type=':
        type = sys.argv[1][7:]
        argv=argv[1:]

    if type == 'help' or not type in types:
        print ('Supported types:')
        for t in types:
            print ('%s - %s' %(t, types[t]['description']))
        sys.exit(1)
    if len(argv) < 2:
        _usage();
        sys.exit(1)
    module = argv[0]
    class_name = argv[1]
    namespace = module
    if len(argv) > 2:
        namespace = argv[2]
    base = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_dir = os.path.join(base,'data')
    module_dir = os.path.abspath(os.path.join(base,'..','..','src','modules'))
    _check('module directory (%s) exists'%module_dir,os.path.exists(module_dir))
    target_dir = os.path.join(module_dir,module)
    _check('target directory  (%s) does not exist'%target_dir,lambda: not os.path.exists(target_dir))
    source = os.path.join(data_dir,types[type]['source'])
    header = os.path.join(data_dir,types[type]['header'])
    cmake = os.path.join(data_dir,types[type]['cmake'])
    t_source = os.path.join(target_dir,'%s.cpp'%class_name)
    t_header = os.path.join(target_dir,'%s.h'%class_name)
    t_cmake = os.path.join(target_dir,'CMakeLists.txt')
    for x in [source, header, cmake]:
        _check('source template (%s) exists'%x,os.path.exists(x))
    
    username = _get_git_value('user.name','Your name')
    email = _get_git_value('user.email', 'your email')
    user = '%s <%s>' %(username, email)
    print ('User: %s'%user)

    print ('')
    print ('Creating module %s represented by class %s in namespace yuri::%s' % (module, class_name, namespace))
    date_today = datetime.date.today().strftime('%d.%m.%Y')
    year_today = datetime.date.today().strftime('%Y')
    guard = class_name.upper()
    mapping = {'module':module,
               'class_name':class_name,
               'namespace': namespace,
               'guard':guard,
               'date':date_today,
               'year':year_today,
               'user':user
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
    
