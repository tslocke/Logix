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
import types
import compiler
from compiler import ast, pycodegen
import traceback
import new
import itertools as itools
import copy

import language
import rootops
from language import Language, OperatorType
from parser import OperatorSyntax
from data import Symbol, getmeta, setmeta, opfields, opargs, flist
import macros

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

lineno = None

# {{{ Hide compiler.ast api issue with Python 2.4
# A function to create an ast.Function that irons out differences
# introduced in Python 2.4
if ast.Function.__init__.im_func.func_code.co_argcount == 7:
    def astFunction(*args):
        return ast.Function(*args)
elif ast.Function.__init__.im_func.func_code.co_argcount  == 9:
    def astFunction(*args):
        return ast.Function(None, *args)
else:
    assert 0, "Incompatible change in compiler.ast API"
# }}}

# {{{ Code generator hacksHack code generator to allow special global names
# These are evil things to do, but it's the shortest-path-to-working

def CodeGenerator_visitGlobalName(self, node):
    self.set_lineno(node)
    self.emit('LOAD_GLOBAL', node.name)

pycodegen.CodeGenerator.visitGlobalName = CodeGenerator_visitGlobalName

class GlobalName(ast.Node):

    def __init__(self, name):
        self.name = name

    def getChildNodes(self):
        return []

    def __repr__(self):
        return "GlobalName(%r)" % self.name

from compiler.symbols import SymbolVisitor

origVisitGlobal = SymbolVisitor.visitGlobal
SymbolVisitor.visitGlobal = lambda self, node, scope, _=None:\
                            origVisitGlobal(self, node, scope)
origVisitAssign = SymbolVisitor.visitAssign
SymbolVisitor.visitAssign = lambda self, node, scope, _=None:\
                            origVisitAssign(self, node, scope)
# }}}

# {{{ def logixglobal(name):
def logixglobal(name):
    return ast.Getattr(ast.Name('logix'), name)
# }}}

# {{{ class CompileError(Exception):
class CompileError(Exception):
    pass
# }}}

# {{{ def hasResult(obj):
def hasResult(obj):
    if isinstance(obj, rootops.PyOp):
        noreturns = ('while', 'for', 'print', 'try',
                     'import', 'limport', 'from',
                     'yield', 'raise', 'assert',
                     'del', 'exec', 'global',
                     'break', 'continue', 'def',
                     '=', '+=', '-=', '*=', '/=', '%=',
                     '**=', '&=', '|=', '^=',
                     'class')
        optoken = obj.__syntax__.token
        if optoken in noreturns:
            return False

        elif optoken == '(':
            return hasResult(obj[0])

        else:
            return True

    elif isinstance(obj, (rootops.deflang, rootops.defop)):
        return False

    else:
        return True
# }}}

# {{{ def assertResult(obj, action):
def assertResult(obj, action):
    if not hasResult(obj):
        raise CompileError("cannot %s: %s (has no value)" % (action, obj))
# }}}

# {{{ def compileGetOp(op):
def compileGetOp(op):
    lang = op.__language__
    opmodname = op.__language__.__module__

    if opmodname == modulename:
        langexpr = GlobalName(lang.__impl__.name)
    else:
        langexpr = ast.Getattr(ast.Subscript(logixglobal('lmodules'),
                                             'OP_APPLY',
                                             [ast.Const(opmodname)]),
                               lang.__impl__.name)

    return ast.Subscript(
        ast.Getattr(ast.Getattr(langexpr, '__impl__'), 'operators'),
        'OP_APPLY',
        [ast.Const(op.__syntax__.token)])
# }}}

# {{{ def funcArgs(*args, **kws):
def funcArgs(argspec):
    if argspec == None:
        return 0, [], []
    
    argnames = []
    defaults = []
    for arg in argspec.elems:
        if len(arg.elems) == 1:
            if len(defaults) != 0:
                raise CompileError("non-keyword arg %s after keyword arg" % arg[0])
            argnames.append(arg[0])
        else:
            argnames.append(arg[0])
            defaults.append(arg[1])

    for d in defaults:
        assertResult(d, "use as argument default")

    varargs = argspec.get("star")
    varkws = argspec.get("dstar")

    flags = ((varargs and ast.CO_VARARGS or 0) +
             (varkws and ast.CO_VARKEYWORDS or 0))

    if varargs:
        argnames.append(varargs)
        
    if varkws:
        argnames.append(varkws)

    for name in argnames:
        if not isinstance(name, str):
            raise CompileError("invalid argument name %r" % name)

    return flags, map(str, argnames), map(topy, defaults)
# }}}

# {{{ def topy(obj):
def topy(obj):
    """The main compiler function - convert a source object to an ast node"""

    # {{{ ln = line number from metadata (also set global \lineno)
    ln = getmeta(obj, 'lineno')
    if ln:
        global lineno
        lineno = ln
    # }}}

    customcompile = getattr(obj, 'pycompile', None)
    if customcompile:
        node = customcompile(*opargs(obj), **opfields(obj))

    elif isinstance(type(obj), OperatorType):
        node = compileOperator(obj)
    else:
        typ = type(obj).__name__
        method = (getattr(objtopy, typ, None) or
                  getattr(objtopy, "_" + typ, None))
        compilefunc = method and method.im_func

        if compilefunc:
            node = compilefunc(obj)
        else:
            raise CompileError("Cannot compile %s (of type %s)" % (obj, typ))

    if ln:
        node.lineno = ln

    return node
# }}}

# {{{ class objtopy:
class objtopy:

    # {{{ def _int(obj):
    def _int(obj):
        return ast.Const(obj)
    # }}}

    # {{{ def _float(obj):
    def _float(obj):
        return ast.Const(obj)
    # }}}

    # {{{ def _long(obj):
    def _long(obj):
        return ast.Const(obj)
    # }}}

    # {{{ def _str(obj):
    def _str(obj):
        return ast.Const(str(obj))
    # }}}

    # {{{ def bool(obj):
    def bool(obj):
        return obj and ast.Const(True) or ast.Const(False)
    # }}}

    # {{{ def _NoneType(obj):
    def _NoneType(obj):
        return ast.Const(None)
    # }}}

    # {{{ def Symbol(obj):
    def Symbol(obj):
        return ast.Name(str(obj))
    # }}}

    # {{{ def list(obj):
    def list(obj):
        for x in obj: assertResult(x, 'use in list')
        return ast.List(map(topy, obj))
    # }}}

    # {{{ def tuple(obj):
    def tuple(obj):
        for x in obj: assertResult(x, 'use in tuple')
        return ast.Tuple(map(topy, obj))
    # }}}

    # {{{ def dict(obj):
    def dict(obj):
        for k,v in obj.items():
            assertResult(k, "use as dict key")
            assertResult(v, "use in dict")

        return ast.Dict([(topy(k), topy(v)) for k,v in obj.items()])
    # }}}

    # {{{ def flist(obj):
    def flist(obj):
        for x in obj: assertResult(x, 'use in flist')

        return ast.CallFunc(ast.Getattr(logixglobal('flist'), 'new'),
                            [ast.List(map(topy, obj.elems)),
                             ast.Dict([(ast.Const(n), topy(v))
                                       for n,v in obj.items()])])
    # }}}

    # {{{ def OperatorType(op):
    def OperatorType(op):
        return compileGetOp(op)
    # }}}
# }}}

# {{{ def compileOperator(op):
def compileOperator(op):
    getOpFunc = ast.Getattr(compileGetOp(op), 'func')
    return compileFunctionCall(getOpFunc, opargs(op), opfields(op), None, None)
# }}}

# {{{ def compileFunctionCall(pyfunc, argxs, kwxs, starArg, dstarArg):
def compileFunctionCall(pyfunc, argxs, kwxs, starArg, dstarArg):
    try:
        for a in itools.chain(argxs, kwxs.values()):
            assertResult(a, 'pass')
    except:
        pass

    if starArg:
        assertResult(starArg, "pass")

    if dstarArg:
        assertResult(dstarArg, "pass")

    argsc = map(topy, argxs)
    keywords = [ast.Keyword(n, topy(v)) for n,v in kwxs.items()]

    return ast.CallFunc(pyfunc, argsc + keywords,
                        starArg and topy(starArg),
                        dstarArg and topy (dstarArg))
# }}}

# {{{ Rootop Compilers
# {{{ compile CallFunction
def compileCallFunction(op, funcx, *argxs, **kwxs):
    assertResult(funcx, "call")
    star = kwxs.get('*')
    if star is not None:
        del kwxs['*']
    dstar = kwxs.get('**')
    if dstar is not None:
        del kwxs['**']
    
    return compileFunctionCall(topy(funcx), argxs, kwxs, star, dstar)
# }}}

# {{{ compile GetRuleClass
def compileGetRuleClass(op, classname):
    return ast.Getattr(logixglobal('parser'), classname)
# }}}

# {{{ compile setlang:
def compileSetlang(self, langx, temp=False):
    assertResult(langx, "set language to")
    if temp:
        # a top-level setlang - set to a temporary language
        res = ast.CallFunc(logixglobal('tmpLanguage'),
                           [topy(langx), GlobalName('__name__'),
                            ast.CallFunc(GlobalName("globals"), [], None, None)])
    else:
        res = topy(langx)
    res.lineno = getmeta(self, 'lineno')
    return res
# }}} 

# {{{ compile defop:
def compileDefop(self, binding, ruledef, smartspace=None, assoc='left',
                 imp=None, lang=None):

    if lang is None:
        raise CompileError("invalid defop")

    assertResult(ruledef, "use in defop")
    assertResult(binding, "use in defop")

    lineno = getmeta(self, 'lineno')

    # {{{ token = extract from ruledef
    def islit(x): return x[0][0] == 'LiteralRule'

    if islit(ruledef):
        token = ruledef[1]
    elif ruledef[0][0] == 'SequenceRule':
        rules = ruledef[1:]
        if len(rules) > 1 and islit(rules[0]):
            token = rules[0][1]
        elif len(rules) > 1 and islit(rules[1]):
            token = rules[1][1]
        else:
            raise CompileError("invalid ruledef")
    else:
        raise CompileError("invalid ruledef")
    # }}}

    if imp:
        if imp['kind'] == 'm':
            funcname = 'macro'
            
        elif imp['kind'] == 'f':
            funcname = 'func'

        else:
            assert 0, "invalid implementation kind: %s" % imp.kind

        impfuncname = 'operator[%s]' % token

        # {{{ impfunc = 
        argflags, argnames, argdefaults = funcArgs(imp.get('args'))
        impfunc = astFunction(impfuncname,
                              argnames,
                              argdefaults,
                              argflags,
                              None, #docstring
                              ast.Return(block(imp['body'], True)))
        impfunc.lineno = lineno
        # }}}
    else:
        funcname = None
        impfuncname = "None"
        impfunc = None

    lastdot = lang.rfind('.')
    if lastdot == -1:
        langexpr = GlobalName(lang)
    else:
        assert 0, "PROBE: Why are we adding an op to an imported language?"
        langname = lang[lastdot+1:]
        langmod = lang[:lastdot]
        langexpr = ast.Getattr(ast.Subscript(logixglobal('lmodules'),
                                             'OP_APPLY',
                                             [ast.Const(langmod)]),
                               langname)

    newOpCall = ast.CallFunc(
        ast.Getattr(ast.Getattr(langexpr, '__impl__'), 'newOp'),
        [ast.Const(token),
         topy(binding),
         topy(ruledef),
         ast.Const(smartspace),
         ast.Const(assoc),
         ast.Const(funcname),
         ast.Name(impfuncname)
         ])

    if impfunc:
        return ast.Stmt([impfunc, newOpCall])
    else:
        return newOpCall
# }}}

# {{{ compile getops:
def compileGetops(self, fromlang, *ops, **kw):
    targetlang = kw.get("lang")
    if targetlang is None:
        raise CompileError("No target language for getops"
                           "(getops in non-exec parse?)")

    assertResult(fromlang, "get operators from")

    lineno = getmeta(self, 'lineno')

    lastdot = targetlang.rfind('.')
    if lastdot == -1:
        targetlangexpr = GlobalName(targetlang)
    else:
        assert 0, ("PROBE: Why are we getting ops into an imported language?")
        langname = targetlang[lastdot+1:]
        langmod = targetlang[:lastdot]
        targetlangexpr = ast.Getattr(ast.Subscript(logixglobal('lmodules'),
                                                   'OP_APPLY',
                                                   [ast.Const(langmod)]),
                               langname)

    if len(ops) == 0:
        # get all operators
        res = ast.CallFunc(ast.Getattr(ast.Getattr(targetlangexpr, '__impl__'),
                                       'addAllOperators'),
                           [topy(fromlang)])
        
    else:
        stmts = [ast.CallFunc(ast.Getattr(ast.Getattr(targetlangexpr, '__impl__'),
                                          'addOp'),
                              [compileGetOp(op)])
                 for op in ops]

        for s in stmts: s.lineno = lineno
        res = ast.Stmt(stmts)

    res.lineno = lineno
    return res
# }}}

# {{{ compile deflang:
def compileDeflang(self, name, parentx, body):
    assert isinstance(name, Symbol)

    if parentx:
        assertResult(parentx, "derive language from")
        parent = topy(parentx)
    else:
        parent = logixglobal('langlang')

    funcname = "#lang:%s" % name

    body = len(body) > 0 and block(body, False).nodes or []

    funcbody = ast.Stmt(
        body
        + [ast.CallFunc(ast.Getattr(ast.Getattr(ast.Name(str(name)),
                                                '__impl__'),
                                    'addDeflangLocals'),
                        [ast.CallFunc(ast.Name('locals'), [])])])
    
    res = ast.Stmt([
        ast.Assign([compilePlace(Symbol(name))],
                   ast.CallFunc(logixglobal('defLanguage'),
                                [ast.Const(str(name)),
                                 parent,
                                 ast.CallFunc(GlobalName("globals"), [])])),
        astFunction(funcname, tuple(), tuple(), 0, None, funcbody),
        ast.CallFunc(ast.Name(funcname), []),
        ast.AssName(funcname, 'OP_DELETE')])
    
    lineno = getmeta(self, 'lineno')
    for n in res.nodes:
        n.lineno = lineno
    res.lineno = lineno
    return res
# }}}

# {{{ compile quote
def compileQuote(self, arg):
    return quote(arg)
# }}}

# {{{ compile opquote
def compileOpQuote(self, op):
    if hasattr(op, '__syntax__'):
        return compileGetOp(op)
    else:
        raise CompileError("not an operator: %s" % op)
# }}}

# {{{ compile escape
def compileEscape(self, *args, **kws):
    raise CompileError("quote-escape outside quote")
# }}}

# {{{ def installRootopCompilers():
def installRootopCompilers():
    rootops.CallFunction.pycompile = compileCallFunction
    rootops.GetRuleClass.pycompile = compileGetRuleClass
    rootops.setlang.pycompile = compileSetlang
    rootops.defop.pycompile = compileDefop
    rootops.getops.pycompile = compileGetops
    rootops.deflang.pycompile = compileDeflang
    rootops.quote.pycompile = compileQuote
    rootops.opquote.pycompile = compileOpQuote
    rootops.escape.pycompile = compileEscape
# }}}
# }}}

# {{{ class OpFuncs:
class OpFuncs:

    # {{{ def continuationOp(self, expr, parts):
    def continuationOp(self, expr, parts):
        "__continue__"
        assertResult(expr, "call or subscript")

        res = topy(expr)

        for part in parts:
            kind = part[-1]
            args = part[:-1]
            if kind == 'call':
                funcArgs = []
                funcKws = {}
                for a in args:
                    if isinstance(a, rootops.PyOp) and a.__syntax__.token == '=':
                        if isinstance(a[0], Symbol):
                            funcKws[str(a[0])] = a[1]
                        else:
                            raise CompileError("invalid keyword arg %r" % a[0])
                    else:
                        funcArgs.append(a)

                star = part.get('star', None)
                dstar = part.get('dstar', None)
                
                res = compileFunctionCall(res, funcArgs, funcKws, star, dstar)

            elif kind == 'subscript':
                for key in args:
                    assertResult(key, "subscript with")
                res = ast.Subscript(res, 'OP_APPLY', map(topy, args))

            elif kind == 'slice':
                start, end, step = args
                assertResult(start, "slice from")
                assertResult(end, "slice to")
                assertResult(step, "slice with step")

                res = ast.Subscript(res, 'OP_APPLY',
                                    [ast.Sliceobj(map(topy, (start, end, step)))])

        return res
    # }}}

    # {{{ def _global(self, *names):
    def _global(self, *names):
        for n in names:
            if not isinstance(n, Symbol):
                raise CompileError("invalid global")
        return ast.Global([str(n) for n in names])
    # }}}

    # {{{ def _exec(self, code, env=None, locals=None):
    def _exec(self, code, env=None, locals=None):
        assertResult(code, "exec")
        if env is not None:
            assertResult(env, "exec in")
            pyenv = topy(env)
        else:
            pyenv = None
        if locals is not None:
            assertResult(locals, "exec in")
            pylocals = topy(locals)
        else:
            pylocals = None
            
        return ast.Exec(topy(code), pyenv, pylocals)
    # }}}

    # {{{ def _del(self, x):
    def _del(self, x):
        return compilePlace(x, 'OP_DELETE')
    # }}}

    # {{{ def _yield(self, x):
    def _yield(self, x):
        assertResult(x, "yield")
        return ast.Yield(topy(x))
    # }}}

    # {{{ def _raise(self, x, y=None, z=None):
    def _raise(self, x=None, y=None, z=None):
        if x is not None:
            assertResult(x, "raise")
            x = topy(x)

        if y is not None:
            assertResult(y, "raise")
            y = topy(y)

        if z is not None:
            assertResult(z, "raise")
            z = topy(z)

        return ast.Raise(x, y, z)
    # }}}

    # {{{ def _is(self, l, r, **kw):
    def _is(self, l, r, **kw):
        assertResult(l, "use in is")
        assertResult(r, "use in is")
        op = kw.get('not') and 'is not' or 'is'
        return ast.Compare(topy(l), [(op, topy(r))])
    # }}}

    # {{{ def _assert(self, test, message=None):
    def _assert(self, test, message=None):
        assertResult(test, "assert")
        # make sure topy is called in lexical order
        t = topy(test)
        if message is not None:
            assertResult(message, "use in assert")
            message = topy(message)
        return ast.Assert(t, message)
    # }}}

    # {{{ def sub(self, l,r):
    def sub(self, l,r=None):
        "-"
        assertResult(l, "use in -")
        if r == None:
            return ast.UnarySub(topy(l))
        else:
            assertResult(r, "use in -")
            return ast.Sub((topy(l),topy(r)))
    # }}}

    # {{{ def mul(self, l,r):
    def mul(self, l,r=None):
        "*"
        if r == None:
            raise CompileError("missing lhs for *")
        else:
            assertResult(l, "use in *")
            assertResult(r, "use in *")
            return ast.Mul((topy(l),topy(r)))
    # }}}

    # {{{ def power(self, l,r):
    def power(self, l,r=None):
        "**"
        if r == None:
            raise CompileError("missing lhs for **")
        else:
            assertResult(l, "use in **")
            assertResult(r, "use in **")
            return ast.Power((topy(l),topy(r)))
    # }}}

    # {{{ def _break(self):
    def _break(self):
        return ast.Break()
    # }}}

    # {{{ def _continue(self):
    def _continue(self):
        return ast.Continue()
    # }}}

    # {{{ def _for(self, place, lst, body):
    def _for(self, place, lst, body, **kw):
        assertResult(lst, "iterate over")

        # ensure topy in lexical order

        pl = compilePlace(place)
        l = topy(lst)
        bod = block(body, False)

        _else = kw.get('else')
        if _else:
            elseblock = block(_else, False)
        else:
            elseblock = None

        return ast.For(pl, l, bod, elseblock)
    # }}}

    # {{{ def _if(self, test, body, elifs, _else):
    def _if(self, test, body, elifs=None, _else=None):
        assertResult(test, "use as if test")

        if elifs:
            for el in elifs:
                assertResult(el['test'], "use as elif test")
            tests = [(test, body)] + [(el['test'], el['body']) for el in elifs]
        else:
            tests = [(test, body)]

        pytests = []
        for t,b in tests:
            pyt = topy(t)
            pyt.lineno = getmeta(t, 'lineno')
            pytests.append( (pyt, block(b, True)) )

        # HACK: For line numbers. Workaround for a test with no metadata
        # (e.g. just a symbol), but this doesn't work for elifs
        if pytests[0][0].lineno is None:
            pytests[0][0].lineno =  getmeta(self, 'lineno')

        if _else: _else = block(_else, True)

        return ast.If(pytests, _else or ast.Const(None))
    # }}}

    # {{{ def _while(self, test, body):
    def _while(self, test, body, **kw):
        assertResult(test, "use as while test")

        # ensure topy in lexical order
        t = topy(test)
        bod = block(body, False)

        _else = kw.get('else')
        if _else is not None:
            _else = block(_else, False)

        return ast.While(t, bod, _else)
    # }}}

    # {{{ def _def(self, name, body, args=None):
    def _def(self, name, body, args=None):
        flags, argnames, defaults = funcArgs(args)

        if len(body) > 0 and type(body[0]) == str:
            doc = body[0]
            body2 = body[1:]
        else:
            doc = None
            body2 = body

        bod = ast.Return(block(body2, True))
        return astFunction(str(name), argnames, defaults, flags, doc, bod)
    # }}}

    # {{{ def _lambda(self, args, body):
    def _lambda(self, body, args=None):
        flags, argnames, defaults = funcArgs(args)

        basel = self.__language__.__impl__

        if not hasResult(body):
            body = basel.getOp("do:")(body, basel.getOp("return")(None))

        return ast.Lambda(argnames, defaults, flags, topy(body))
    # }}}

    # {{{ def list(self, kind, items):
    def list(self, *args, **kws):
        "["
        def _list(*elems):
            if elems == None:
                return ast.List([])
            else:
                for x in elems: assertResult(x, "use in list literal")

            return ast.List(map(topy,elems))

        def comp(elem, quals):
            assertResult(elem, 'use in list comprehension')

            def qual(place, list, ifs=None):
                ifs = ifs or []
                assertResult(list, "generate list from")
                for test in ifs: assertResult(test, "filter list with")
                return ast.ListCompFor(compilePlace(place),
                                       topy(list),
                                       [ast.ListCompIf(topy(test))
                                        for test in ifs])

            return ast.ListComp(topy(elem), [qual(*q.elems, **q.fields)
                                             for q in quals])

        kind = args[-1]
        els = args[:-1]
        if kind == 'list':
            return _list(*els, **kws)
        if kind == 'slice':
            return slice(*els, **kws)
        if kind == 'comp':
            return comp(*els, **kws)

        assert 0
    # }}}

    # {{{ def dict(self, *items):
    def dict(self, *items):
        "{"

        if items == (None,):
            return ast.Dict([])
        else:
            for x in items[1::2]: assertResult(x, "use in dict literal")

            return ast.Dict([(topy(items[i]), topy(items[i+1]))
                             for i in range(0, len(items), 2)])
    # }}}

    # {{{ assign = 
    def assign(self, place, val):
        "="
        if isinstance(val, self.__class__):
            # chained assign (a = b = blah)
            res = topy(val)
            res.nodes.append(compilePlace(place))
            return res
        else:
            assertResult(val, "assign from")
            res = ast.Assign([compilePlace(place)], topy(val))
            res.lineno = lineno
            return res
    # }}}

    # {{{ def _print(self, dest, *args, **kws):
    def _print(self, dest, *args, **kws):
        node = kws.get('nonl') and ast.Print or ast.Printnl

        for arg in args: assertResult(arg, "print")

        asts = [topy(x) for x in args if x != None]
        if dest == "tofile":
            out = asts[0]
            prn = asts[1:]
        else:
            out = None
            prn = asts

        return node(prn, out)
    # }}}

    # {{{ def _return(self, val):
    def _return(self, val):
        if val is None:
            pyval = ast.Const(None)
        else:
            assertResult(val, "return")
            pyval = topy(val)
        return ast.Return(pyval)
    # }}}

    # {{{ def paren(self, expr):
    def paren(self, expr):
        "("
        if expr == None:
            raise CompileError("empty parentheses")
        return topy(expr)
    # }}}

    # {{{ def do(self, *body):
    def do(self, *body):
        "do:"
        return block(body, True)
    # }}}

    # {{{ def semi(self, *stmts):
    def semi(self, *stmts):
        ";"
        return block(stmts, True)
    # }}}

    # {{{ def _import(self, module, as):
    def _import(self, module, as=None):
        if as != None and not isinstance(as, Symbol):
            raise CompileError("invalid 'as' clause in input")
        return ast.Import([('.'.join(module), as and str(as))])
    # }}}

    # {{{ def limport(self, module, as):
    def limport(self, module, as=None):
        if as != None:
            if not isinstance(as, Symbol):
                raise CompileError("invalid 'as' clause in input")
        else:
            as = module[-1]
        fname = '.'.join(module)
        return ast.Assign([ast.AssName(str(as), 'OP_ASSIGN')],
                          ast.CallFunc(logixglobal('imp'),
                                       [ast.Const(fname),
                                        ast.CallFunc(GlobalName('globals'), [])]))
    # }}}

    # {{{ def dot(self, expr, field):
    def dot(self, expr, field):
        "."
        assertResult(expr, "get attribute from")
        if isinstance(field, Symbol):
            return ast.Getattr(topy(expr), str(field))
        else:
            raise CompileError("object field must be a symbol (got %s)" % field)
    # }}}

    # {{{ def comma(*elems):
    def comma(self, *elems):
        ","
        for e in elems: assertResult(e, "use in tuple")
        return ast.Tuple(map(topy, elems))
    # }}}

    # {{{ def _not(self, x, elem=None, **kw):
    def _not(self, x, elem=None, **kw):
        if 'in' in kw:
            assertResult(x, "test for membership")
            assertResult(elem, "test for membership")
            return ast.Compare(topy(elem), [('not in', topy(x))])
        else:
            assertResult(x, "negate")
            if elem != None:
                raise CompileError("invalid 'not'")
            return ast.Not(topy(x))
    # }}}

    # {{{ def invert(self, e):
    def invert(self, e):
        "~"
        assertResult(e, "invert")
        return ast.Invert(topy(e))
    # }}}

    # {{{ def _class(self, name, bases, body):
    def _class(self, name, body, bases=None, ):
        if not isinstance(name, Symbol):
            raise CompileError("invalid class name")

        if bases:
            for b in bases: assertResult(b, "use as base class")
        else:
            bases = []

        return ast.Class(str(name), map(topy, bases), None, block(body, False))
    # }}}

    # {{{ def _try(self, body, kind, **kw):
    def _try(self, body, **kw):
        excepts = kw.get('excepts')
        if excepts:
            def one(handler, exc=None, target=None):
                assertResult(exc, "trap exception")
                excc = exc and topy(exc)
                targetc = target and compilePlace(target)
                return (excc, targetc, block(handler, False))

            _else = kw.get('else')
            if _else is not None:
                _else = block(_else, False)

            return ast.TryExcept(block(body, False),
                                 [one(*e.elems, **e.fields) for e in excepts],
                                 _else)

        else:
            return ast.TryFinally(block(body, False),
                                  block(kw['finally'], False))
    # }}}

    # {{{ def _from(self, module, imp):
    def _from(self, module, imp):
        if imp == 'star':
            imps = [('*', None)]
        else:
            def tostr(a):
                if a is None: return None
                else: return str(a)
            imps = [(tostr(a), tostr(b)) for a,b in zip(imp[::2], imp[1::2])]

        return ast.From(str('.'.join(module)), imps)
    # }}}

    # {{{ def string(self, s):
    def string(self, s):
        '"""'
        return ast.Const(eval('"""' + s + '"""'))
    # }}}
# }}}

# {{{ def installPyCompilers(basel):
def installPyCompilers(basel):

    # {{{ string literals
    allQuotes = [ (u + r, q) for u in '', 'u', 'U'
                             for r in '', 'r', 'R'
                             for q in "'", '"', "'''", '"""']

    for mod, q in allQuotes:
        def one(mod, q):
            def makeStr(self, text):
                try:
                    return ast.Const(eval(mod + q + text + q))
                except SyntaxError, e:
                    raise CompileError, str(e)

            basel.getOp(mod + q).pycompile = makeStr
        one(mod, q)
    # }}}

    # {{{ binops = 
    binops = {
        '+':   ast.Add,
        '/':   ast.Div,
        '//':  ast.FloorDiv,
        '%':   ast.Mod,
        'and': ast.And,
        'or':  ast.Or,
        '>>':  ast.RightShift,
        '<<':  ast.LeftShift,
        '&':  ast.Bitand,
        '|':  ast.Bitor,
        '^':  ast.Bitxor,
        }

    for k,v in binops.items():
        def one(k,v):
            def makeAST(self, l,r):
                assertResult(l, "use in "+k)
                assertResult(r, "use in "+k)
                return v((topy(l),topy(r)))

            basel.getOp(k).pycompile = makeAST
        one(k,v)
    # }}}

    # {{{ assignops = 
    assignops = [
        '+=',
        '-=',
        '*=',
        '/=',
        '%=',
        '**=',
        '&=',
        '|=',
        '^=',
        ]

    for op in assignops:
        def one(op):
            def makeAST(self, place, val):
                # check it's a valid place (discard result)
                compilePlace(place) 

                assertResult(val, "use in %s" % op)
                return ast.AugAssign(topy(place),
                                     op,
                                     topy(val))
            basel.getOp(op).pycompile = makeAST

        one(op)
    # }}}

    # {{{ compareops = 
    compareops = [ '==', '!=', '>', '<', '>=', '<=', 'in', '<>']

    for op in compareops:
        def one(o):
            def makeAST(op, l,r):
                assertResult(l, "use in " + o)
                assertResult(r, "use in " + o)
                op = o == '<>' and '!=' or o
                return ast.Compare(topy(l), [(op, topy(r))])
            basel.getOp(o).pycompile = makeAST
        one(op)
    # }}}

    # {{{ opfuncs
    for name, func in OpFuncs.__dict__.items():
        if name in ('__module__', '__doc__'): continue

        if func.__doc__ != None: name = func.__doc__
        elif name.startswith("_"): name = name[1:]
        basel.getOp(name).pycompile = func
    # }}}
# }}}

# {{{ def compilePlace(place, opname):
def compilePlace(place, opname='OP_ASSIGN'):
    if isinstance(place, Symbol):
        return ast.AssName(str(place), opname)

    elif isinstance(place, rootops.PyOp):
        token = place.__syntax__.token

        if token == ".":
            # {{{ assign to field
            expr = topy(place[0])
            field = place[1]
            if isinstance(field, Symbol):
                return ast.AssAttr(expr, str(field), opname)
            else:
                raise CompileError("Cannot assign to %s" % place)
            # }}}

        elif token == "__continue__":
            # {{{ assign to slice or subscript
            last = place[1][-1]

            # expr = the continuationOp not including the last part
            if len(place[1]) == 1:
                expr = place[0]
            else:
                expr = basel.getContinuationOp()(place[0], place[1][:-1])
            
            kind = last[-1]
            if kind == 'subscript':
                # {{{ assign to subscript
                assertResult(expr, "subscript")

                keys = last[:-1]
                for key in keys:
                    assertResult(key, "subscript with")

                return ast.Subscript(topy(expr), opname, map(topy, keys))
                # }}}

            elif kind ==  "slice":
                # {{{ assign to slice
                assertResult(expr, "slice")

                start,end,step = last[:-1]
                assertResult(start, "slice from ")
                assertResult(end, "slice to")
                assertResult(step, "slice with step")

                return ast.Subscript(topy(expr), opname,
                                     [ast.Sliceobj(map(topy, (start,end,step)))])
                # }}}
                
            else:
                raise CompileError("Cannot asign to %s " % place)
            # }}}

        elif token == ",":
            return ast.AssTuple([compilePlace(p, opname) for p in list(place)])

        elif token == "[" and place[1] == 'list':
            return ast.AssList([compilePlace(p, opname) for p in place.elems])

        elif token == "(":
            return compilePlace(place[1], opname)

        else:
            raise CompileError("Cannot asign to %s " % place)

    else:
        raise CompileError("Cannot assign to %s" % place)
# }}}

# {{{ def block(recl, withResult, prepend=None):
def block(lst, withResult, prepend=None):
    if len(lst) == 0:
        if withResult:
            return ast.Const(None)
        else:
            return ast.Pass()

    def stm(x):
        if hasResult(x):
            dis = ast.Discard(topy(x))
            dis.lineno = getmeta(x, 'lineno')
            return dis
        else:
            return topy(x)

    if withResult:
        stms = [stm(s) for s in lst[:-1]]
        last = lst[-1]
        lastpy = topy(last)

        if hasResult(last):
            # Insert a pass so pycodegen emits a SET_LINENO
            p = ast.Pass()
            p.lineno = getmeta(last, 'lineno')
        
            statements = stms + [p, lastpy]
        else:
            statements = stms + [lastpy, ast.Const(None)]
    else:
        statements = [stm(s) for s in lst]

    if prepend:
        return ast.Stmt(prepend + statements)
    else:
        return ast.Stmt(statements)
# }}}

# {{{ def compile(src, filename, mode, showTree=False):
def compile(src, filename, mode='exec', showTree=False, importlogix=True,
            module=None):
    global lineno
    lineno = 0
    global modulename
    modulename = module

    implogix = ast.Import([(logixModuleName, 'logix')])
    prepend = importlogix and [implogix] or None

    if len(src) == 0:
        src = [None]

    try:
        if mode == "exec":
            statements = block(src, False, prepend)
            tree = ast.Module(None, statements)
            gen = pycodegen.ModuleCodeGenerator

        else:
            assert len(src) == 1
            stmt = src[0]

            if mode == "single":
                statements = block([stmt], False, prepend=prepend)
                tree = ast.Module(None, statements)
                gen = pycodegen.InteractiveCodeGenerator

            elif mode == "eval":
                statements = block([stmt], True, prepend)
                tree = ast.Expression(statements)
                gen = pycodegen.ExpressionCodeGenerator
            else:
                raise ValueError("compile() 3rd arg must be 'exec' or "
                                 "'eval' or 'single'")
    except CompileError, exc:
        offset = None
        raise SyntaxError(str(exc), (filename, lineno, offset, None))

    compiler.misc.set_filename(filename, tree)
    compiler.syntax.check(tree)

    if showTree: print tree

    code = gen(tree).getCode()

    del lineno
    
    return code
# }}}

# {{{ def dump_module(src, modname, file):
def dump_module(src, filename, file):
    code = compile(src, modname + '.lx', mode='exec')
    savemodule(code, modname + '.lxc', modname + 'lx')
    return code
# }}}

# {{{ def compileCodeObjects(filename, codeobjs):
def compileCodeObjects(filename, codeobjs):
    if len(codeobjs) == 0:
        stmts = []
    else:
        stmts = [ast.For(ast.AssName('[--codeobj--]', 'OP_ASSIGN'),
                         ast.Const(codeobjs),
                         ast.Stmt([ast.Exec(ast.Name('[--codeobj--]'),
                                            None, None)]),
                         None),
                 ast.AssName('[--codeobj--]', 'OP_DELETE')]

    module = ast.Module(None, ast.Stmt(stmts))
    compiler.misc.set_filename(filename, module)
    return pycodegen.ModuleCodeGenerator(module).getCode()
# }}}

# {{{ def quote(obj):
def localModuleQuote():
    return rootops.quote(rootops.localmodule(rootops.escape(Symbol("__name__"),
                                                            extra=[])))

# {{{ def quotedArgs(operands, depth):
def quotedArgs(operands, depth):
    """
    Generate code from an flist of operand expressions, possibly containing splices.
    
    Returns an expression that constructs an flist.
    """

    parts = []
    for x in operands.elems:
        if isinstance(x, rootops.escape):
            extra = x.__operands__.get('extra', [])
            escapelevel = 1 + len(extra)
            if escapelevel > depth:
                raise CompileError("invalid quote escape")
            escape = escapelevel == depth
        else:
            escape = False
            
        if escape:
            if x.__operands__.hasField("splice"):
                assertResult(x[0], "insert into quote")
                parts.append( ('s', topy(x[0])) )  # 's' == splice
            
            elif x.__operands__.hasField("localmodule"):
                parts.append( ('v',topy(localModuleQuote())) )
                     
            else:
                assertResult(x[0], "insert into quote")
                parts.append( ('v', topy(x[0])) )  # 'v' == plain value
            
        else:
            parts.append( ('v', quote(x, depth)) )  # 'v' == plain value

    # {{{ expr = reduce parts to a single expression
    # If there is just a single splice, then that is the expression
    # otherwise generate:  val = ??; val.extend(??); val.extend(??)...
    def frontSection(parts):
        vals = list(itools.takewhile(lambda (tag, exp): tag == 'v', parts))
        if len(vals) == 0:
            return parts[0][1], parts[1:]
        else:
            return ast.List([v[1] for v in vals]), parts[len(vals):]
        
    if len(parts) == 0:
        expr = ast.List([])
    else:
        first, rest = frontSection(parts)
        if len(rest) == 0:
            expr = first
        else:
            # Generate:
            #     val = ...; if not hasattr(val, 'extend'): val = list(val)
            val = macros.gensym("val")
            statements = [
                ast.Assign([compilePlace(val)], first),
                ast.If([(ast.CallFunc(GlobalName('isinstance'),
                                      [topy(val), logixglobal("flist")]),
                         ast.Assign([compilePlace(val)],
                                    ast.CallFunc(ast.Getattr(topy(val), "copy"),
                                                 [])))],
                       #else
                       ast.Assign([compilePlace(val)],
                                  ast.CallFunc(GlobalName('list'), [topy(val)])))]

            while len(rest) > 0:
                ex, rest = frontSection(rest)

                statements.append(ast.Discard(ast.CallFunc(ast.Getattr(topy(val),
                                                                       "extend"),
                                                           [ex])))
            statements.append(topy(val))

            expr = ast.Stmt(statements)
    # }}}

    for v in operands.fields.values():
        assertResult(v, "use as operand")
        
    keywords = ast.Dict([(ast.Const(n), quote(v, depth))
                         for n, v in operands.items()])

    if isinstance(expr, ast.List):
        return ast.CallFunc(ast.Getattr(logixglobal("flist"), 'new'),
                            [expr, keywords])
    else:
        return ast.Add([expr, ast.CallFunc(ast.Getattr(logixglobal("flist"), 'new'),
                                           [ast.List([]), keywords])])
# }}}

def quote(obj, depth=1):
    assert depth > 0

    if isinstance(obj, rootops.quote):
        depth += 1

    elif isinstance(obj, rootops.escape):
        extra = obj.__operands__.get('extra', [])
        escapelevel = 1 + len(extra)
        if escapelevel > depth:
            raise CompileError("invalid quote escape")
        
        elif escapelevel == depth:
            if obj.__operands__.hasField('splice'):
                raise CompileError("Can't splice here")

            elif obj.__operands__.hasField("localmodule"):
                return topy(localModuleQuote())
            
            else:
                assertResult(obj[0], "insert into quote")
                return topy(obj[0])

    if isinstance(type(obj), OperatorType):
        # {{{ generate code to build the operator
        cls = obj.__class__
        if cls.__module__ == '%s.rootops' % logixModuleName:
            classx = ast.Getattr(logixglobal('rootops'), cls.__name__)
        else:
            optoken = obj.__class__.__syntax__.token
            classx = compileGetOp(obj)

        operands = macros.gensym("operands")
        return ast.Stmt([ast.Assign([compilePlace(operands)],
                                    quotedArgs(obj.__operands__, depth)),
                         ast.CallFunc(classx, [],
                                      ast.Getattr(topy(operands), "elems"),
                                      ast.Getattr(topy(operands), "fields"))])
        # }}}

    elif isinstance(obj, flist):
        # {{{ generate code to build the flist
        return quotedArgs(obj, depth)
        # }}}

    elif isinstance(obj, Symbol):
        # {{{ generate code to build the symbol
        return ast.CallFunc(logixglobal('Symbol'), [ast.Const(obj.asStr())])
        # }}}

    elif isinstance(obj, (tuple, list, dict)):
        # Q: Is this ok?
        assert 0, "didn't expect one of those to be quoted"

    else:
        return topy(obj)
# }}}
