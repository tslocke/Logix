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
import sys

import pycompile
logixcompiler = pycompile
import language
import macros
import rootops

try:
    from livedesk.util import debug, debugmode
    import mydb
except ImportError:
    pass
# }}}

# Warning: herein lie some horrific hacks to make the prompt change dynamically :-)

# {{{ def initInteractive(ipy, baselang):
def initInteractive(ipy, baselang):
    package = __name__[:__name__.rfind(".")]
    logix = sys.modules[package]

    ipy.locals['logix'] = logix

    global _ipy
    _ipy = ipy

    logix.init()
    
    # {{{ optimisations
    if 0:
        import psyco
        import compiler
        psyco.bind(language.Language)
        psyco.bind(compiler.pycodegen, rec=10)

    if 0:
        import optimize
        import compiler
        optimize.bind_all(compiler)
        optimize.bind_all(parser)
    # }}}

    if baselang:
        ipy.locals['__currentlang__'] = logix.baselang
    else:
        mod = logix.imp('%s.std' % package, {})
        tmpstd = language.tmpLanguage(mod.std, '__main__')
        ipy.locals['__currentlang__'] = tmpstd
        ipy.locals[tmpstd.__impl__.name] = tmpstd
        logix.stdlang = mod.std

    print "(type logix.license for more information)"
    print "Welcome to Logix"
# }}}

# {{{ def logix_raw_input(prompt=''):
def logix_raw_input(prompt=''):
    if prompt is _ipy.outputcache.prompt1:
        _setprompt()
        prompt = _ipy.outputcache.prompt1
    return _ipy.standard_raw_input(prompt)
# }}}

# {{{ def icompile()
def icompile(src, filename='<input>', mode='single'):
    import sys

    if src.strip() == "":
        return compile("None", "", "eval")

    # {{{ compile ipython magic commands as Python code
    if src.startswith("__IP.magic"):
        import codeop
        return codeop.CommandCompiler()(src, filename, mode)
    # }}}
    
    # {{{ return None if src is unfinished multiline
    if src.count("\n") > 0:
        if not src.endswith("\n"):
            return None
    else:
        if src.strip().endswith(":"):
            return None
    # }}}

    import IPython
    AutoTB = IPython.ultraTB.AutoFormattedTB(mode='Context', color_scheme='Linux',
                                             call_pdb=1)

    try:
        ulang = _ipy.locals['__currentlang__']

        if not isinstance(ulang, language.Language):
            raise SyntaxError("current language is not valid")
        elif not ulang.__impl__.name.endswith("~"):
            ulang = language.tmpLanguage(ulang, _ipy.locals['__name__'])
            _ipy.locals['__currentlang__'] = ulang
        
        lang = ulang.__impl__
        
        expr = lang.parse(src, mode="interactive", execenv=_ipy.locals)

        return interactiveCompile(expr, _ipy.locals, filename, mode)
    
    # {{{ exception handling
    except SyntaxError:
        raise
    except macros.MacroExpandError, exc:
        macros.formatMacroError(exc)
        exc_info = exc.exc_info
        raise ValueError("macro expand error")

    except:
        AutoTB()
        raise ValueError, sys.exc_info()[1]
    # }}}
# }}}

# {{{ def _setprompt():
def _setprompt():
    import IPython.Prompts

    try:
        n = _ipy.locals['__currentlang__'].__impl__.name
        if n.endswith ("~"):
            langname = n[:-1]
        else:
            langname = n
    except:
        langname = 'no language!'
    prompt = '[%s]:' % langname
    cprompt = '[${self.col_num}%s${self.col_p}]:' % langname
    
    p = IPython.Prompts.Prompt1(_ipy.outputcache, prompt=cprompt)
    p.set_colors()
    p.p_str_nocolor = prompt
    
    _ipy.outputcache.prompt1 = p
# }}}

# {{{ def toggleLogixMode(ip, baselang=False):
def toggleLogixMode(ip, baselang=False):
    if ip.rc.autocall:
        print "(switching off IPython autocall feature)"
        print
        ip.rc.autocall = False

    if not globals().has_key('_ipy'):
        ip.standard_prompt = ip.outputcache.prompt1
        ip.standard_compile = None
        initInteractive(ip, baselang)

    if ip.standard_compile == None:
        ip.standard_compile = ip.compile
        ip.compile = icompile
        ip.standard_raw_input = ip.raw_input
        ip.raw_input = logix_raw_input
    else:
        ip.compile = ip.standard_compile
        ip.standard_compile = None
        ip.raw_input = ip.standard_raw_input
        ip.outputcache.prompt1 = ip.standard_prompt
# }}}
    
# {{{ def interactiveCompile(expr, env, filename, mode)
# This doesn't really belong here because it is also used by the non-IPython prompt
# but where else would it go?

def interactiveCompile(expr, env, filename, mode):
    modname = env['__name__']
    if isinstance(expr, rootops.setlang):
        newlang = language.tmpLanguage(language.eval(expr, env), modname)
        env['__currentlang__']  = newlang
        env[newlang.__impl__.name] = newlang
        return compile("None", "", "eval")

    if isinstance(expr, (rootops.defop, rootops.getops)):
        expr['lang'] = '__currentlang__'
        language.eval(expr, env)
        return compile("None", "", "eval")
    else:
        return logixcompiler.compile([expr], filename, mode, module=modname)
# }}}
