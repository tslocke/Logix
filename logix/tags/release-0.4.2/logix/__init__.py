# {{{ GPL License
# Logix - an extensible programming language for Python
# Copyright (C) 2004 Tom Locke
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# (http://www.gnu.org/copyleft/gpl.html)
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# }}}

# {{{ imports
import os
import os.path
import sys
modules = sys.modules

try:
    from livedesk.util import debug
except ImportError:
    pass

from data import Symbol, opargs, opfields, getmeta, flist
from language import Language, LanguageImpl, eval, \
                     OperatorType, BaseMacro, tmpLanguage, defLanguage
from macros import MacroExpandError, gensym, formatMacroError
from parser import OperatorSyntax

import rootops
import pycompile
logixcompiler = pycompile
logixcompiler.logixModuleName = __name__
import language
import parser
import macros

CallFunction = rootops.CallFunction
# }}}

# {{{ globals
lmodules = {'__main__':sys.modules['__main__']}
home = __file__[:__file__.rfind(os.path.sep)]
# }}}

# {{{ def init():
def init():
    logixModuleName = __name__
    global quotelang, syntaxlang, langlang, baselang
    quotelang, syntaxlang, langlang, baselang = \
               rootops.makeLanguages(logixModuleName, home, lmodules)

    global stdlang
    stdlang = imp("%s/std" % logixModuleName).std
# }}}

# {{{ def macroexpand(x):
def macroexpand(x):
    return macros.expand(x)
# }}}

# {{{ def macroexpand1(x, context=None):
def macroexpand1(x, context=None):
    if context is None:
        context = MacroContext()
    return macros.expand1(x, context)
# }}}

# {{{ def topython(x):
def topython(x):
    return logixcompiler.topy(macroexpand(x))
# }}}

# {{{ def loadmodule(filename):
def loadmodule(filename):
    import marshal, new

    cfile = filename + ".lxc"
    f = file(cfile, "rb")
    f.read(8)
    code = marshal.load(f)
    f.close()

    mod = new.module(filename)
    mod.__file__ = os.path.abspath(cfile).replace('/', '.')
    lmodules[filename.replace('/', '.')] = mod

    exec code in vars(mod)
    
    return mod
# }}}
    
# {{{ def imp(modulename, globals=None):
def imp(modulename, globals=None):
    import sys

    if globals is None:
        globals = sys._getframe(-1).f_globals
    
    for p in sys.path:
        mod = _imp(p, modulename, globals)
        if mod is not None:
            return mod
    raise ImportError("No logix module named %s" % modulename)
    
def _imp(path, modulename, globals):
    import marshal, new, os
    mtime = os.path.getmtime
    exists = os.path.exists

    # HACK: To make limport work in devlogix lmodules
    if __name__ == 'devlogix' and modulename.startswith("logix."):
        modulename = 'dev' + modulename

    # {{{ if already in lmodules, just return it
    current = lmodules.get(modulename)
    if current:
        return current
    # }}}

    basename = modulename.replace(".","/")

    # might need to look for it within a package
    if (not exists(os.path.join(path, basename + ".lx"))
        and '__name__' in globals
        and '__file__' in globals):
        # {{{ modulename = package local name
        importer = globals['__name__']
        if globals['__file__'].endswith("__init__.py"):
            package = importer
        else:
            idx = importer.rfind('.')
            if idx == -1:
                package = importer
            else:
                package = importer[:idx]
        modulename = package + '.' + modulename
        # }}}
        basename = modulename.replace(".","/")
        # {{{ try again - if already in lmodules, just return it
        current = lmodules.get(modulename)
        if current:
            return current
        # }}}

    sfile = os.path.join(path, basename + ".lx")
    cfile = os.path.join(path, basename + ".lxc")

    if exists(sfile):
        # {{{ ensure package is loaded
        doti = modulename.rfind('.')
        if doti != -1:
            __import__(modulename[:doti], globals, None)
        # }}}
        # {{{ mod = create the module
        #print "limport", modulename
        mod = new.module(modulename)
        mod.__name__ = modulename
        mod.__file__ = cfile

        lmodules[modulename] = mod

        try:
            if not exists(cfile) or mtime(sfile) > mtime(cfile):
                execmodule(sfile, vars(mod))
            else:
                f = file(cfile, "rb")
                f.read(8)
                code = marshal.load(f)
                f.close()
                exec code in vars(mod)

                
        except MacroExpandError, e:
            print 'removing module', modulename
            del lmodules[modulename]
            formatMacroError(e, sfile)
            raise
        except:
            print 'removing module', modulename
            del lmodules[modulename]
            import traceback
            traceback.print_exc()
            raise
        # }}}
        return mod
    
    else:
        return None
# }}}

# {{{ def makecompile(src, fname, mode): DISABLED
def XXmakecompile(modulename):
    def compile(src, fname, mode):
        "compatible replacement for builtin compile()"

        recls = baselang.__impl__.parse(src, filename=fname,
                                        execenv=dict(__name__=modulename))
        return logixcompiler.compile(recls, fname, mode)
    return compile
# }}}

# {{{ def execfile(filename, globals=None):
def execfile(filename, globals=None):
    execenv = globals or {}
    baselang.__impl__.parse(file(filename, 'U').read(), filename=filename,
                            execenv=execenv)
# }}}
    
# {{{ def execmodule(filename, globs):
def execmodule(filename, globs):
    import time
    now = time.time()
    cfile = filename + "c"

    if "baselang" not in globals():
        init()
    code = baselang.__impl__.parse(file(filename, 'U'), filename,
                                   mode='execmodule', execenv=globs)
    print "built %s in %f" % (filename, time.time() - now)
    savemodule(code, cfile, filename)
# }}}

# {{{ def clearCaches():
def clearCaches():
    LanguageImpl.clearAllCaches()
# }}}

# {{{ class ArgPreconditionError(Exception):
class ArgPreconditionError(Exception):

    def __init__(self, funcname, argname, argval):
        self.argval = argval
        self.msg = "%s: arg-precondition failed for '%s' (got %r)" % \
            (funcname, argname, argval)

    def __str__(self): return self.msg
# }}}
                           
# {{{ def savemodule(code, filename):
def savemodule(code, filename, srcfile):
    # pinched from compiler.pycodegen.Module.getPycHeader
    import struct
    import imp
    import marshal
    
    mtime = os.path.getmtime(srcfile)
    mtime = struct.pack('<i', mtime)
    header = imp.get_magic() + mtime

    f = file(filename, "wb")
    f.write(header)
    marshal.dump(code, f)
    f.close()
# }}}

# {{{ class LogixConsole(code.InteractiveConsole):
import code
class LogixConsole(code.InteractiveConsole):

    def __init__(self):
        code.InteractiveConsole.__init__(self)
        import sys

    def runsource(self, src, filename='<input>', mode='single'):
        import sys

        if src.strip() == "":
            return False

        # {{{ return True if src is unfinished multiline
        if src.count("\n") > 0:
            if not src.endswith("\n"):
                return True
        else:
            if src.strip().endswith(":"):
                return True
        # }}}

        try:
            ulang = self.locals['__currentlang__']

            if not isinstance(ulang, language.Language):
                raise SyntaxError("current language is not valid")
            elif not ulang.__impl__.name.endswith("~"):
                ulang = language.tmpLanguage(ulang, self.locals['__name__'])
                self.locals['__currentlang__'] = ulang
            
            lang = ulang.__impl__
            
            expr = lang.parse(src, mode='interactive', execenv=self.locals)

            import ipy
            code = ipy.interactiveCompile(expr, self.locals, filename, mode)
            self.runcode(code)

            # {{{ update prompt with current language name
            lang = self.locals['__currentlang__']
            name = lang.__impl__.name
            sys.ps1 = "[%s]: " % (name.endswith("~") and name[:-1] or name)
            # }}}
            return False

        # {{{ exception handling
        except SyntaxError:
            self.showsyntaxerror(filename)
        except MacroExpandError, exc:
            formatMacroError(exc)
    # }}}

    def interact(self, *args):
        if hasattr(sys, 'ps1'):
            old_prompt = sys.ps1
        else:
            old_prompt = None
        import language
        stdtmp = language.tmpLanguage(stdlang, self.locals['__name__'])
        self.locals['__currentlang__'] = stdtmp
        self.locals[stdtmp.__impl__.name] = stdtmp
        sys.ps1 = "[std]: "
        try:
            code.InteractiveConsole.interact(self, *args)
        finally:
            if old_prompt:
                sys.ps1 = old_prompt
# }}}

# {{{ def interact():
def interact():
    init()
    global stdlang
    stdlang = imp("%s.std" % __name__).std

    print "(type logix.license for more information)"
    
    LogixConsole().interact("Welcome to Logix")
# }}}

def parse(language, src, filename=None, mode='parse', execenv=None):
    return language.__impl__.parse(src, filename, execenv, mode)

# {{{ interactive license message
class License:
    def __repr__(self): return """
Logix is 'free software'. You are licensed to use it under the terms
of the GNU General Public License. Logix comes with NO WARRANTY.
For more information see the file LICENSE.TXT that is distributed
along with this software or visit http://www.gnu.org/copyleft/gpl.html
"""

license = License()
del License

# }}}
