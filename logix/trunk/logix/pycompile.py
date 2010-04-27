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
from language import Language
from parser import OperatorSyntax
import data
from data import Symbol, Doc, isDoc
import macros

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

lineno = None

rootopCompilers = {}

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

# {{{ Code generator hacks
# These are evil things to do, but it's the shortest-path-to-working

# Hack code generator to allow special global names
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

# {{{ def isPyOp(x):
def isPyOp(x, op=None):
    if isinstance(x, Doc) and x.tag.namespace == rootops.base_ns:
        return op is None or x.tag.name == op
    else:
        return False
# }}}

# {{{ def hasResult(obj):
def hasResult(obj):
    if isPyOp(obj):
        nores = ('while', 'for', 'print', 'try',
                 'import', 'limport', 'from',
                 'yield', 'raise', 'assert',
                 'del', 'exec', 'global',
                 'break', 'continue', 'def',
                 '=', '+=', '-=', '*=', '/=', '%=',
                 '**=', '&=', '|=', '^=',
                 'class')
        optoken = obj.tag.name
        if optoken in nores:
            return False

        elif optoken == '(':
            return hasResult(obj[0])

        else:
            return True

    elif isinstance(obj, Doc) and obj.tag in (rootops.deflang, rootops.defop):
        return False

    else:
        return True
# }}}

# {{{ def assertResult(obj, action):
def assertResult(obj, action):
    if not hasResult(obj):
        raise CompileError("cannot %s: %s (has no value)" % (action, obj))
# }}}

# {{{ def compileGetOp(symbol):
def compileGetOp(symbol):
    lang = language.getLanguage(symbol.namespace)

    if lang is None:
        debug()
        raise CompileError, "not an operator symbol: %s" % symbol
        
    opmodname = lang.__module__

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
        [ast.Const(symbol.name)])
# }}}

# {{{ def funcArgs(*args, **kws):
def funcArgs(argspec):
    if argspec == None:
        return 0, [], []
    
    argnames = []
    defaults = []
    for arg in argspec:
        if len(arg) == 1:
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
        if not isinstance(name, Symbol):
            raise CompileError("invalid argument name: %r" % name)

    return flags, map(str, argnames), map(topy, defaults)
# }}}

# {{{ def topy(obj):
def topy(obj):
    """The main compiler function - convert a code-doc into an ast node"""

    # {{{ ln = line number from metadata (also set global \lineno)
    global lineno
    ln = getattr(obj, 'lineno', None)
    if ln != None:
        lineno = ln
    # }}}

    typ = type(obj).__name__
    method = (getattr(objtopy, typ, None) or
              getattr(objtopy, "_" + typ, None))
    compilefunc = method and method.im_func

    if compilefunc:
        node = compilefunc(obj)
    else:
        debug()
        raise CompileError("Cannot compile %s (of type %s)" % (obj, typ))

    if ln:
        node.lineno = ln

    return node
# }}}

# {{{ class objtopy:
class objtopy:

    # {{{ def _int(obj):
    def _int(i):
        return ast.Const(i)
    # }}}

    # {{{ def _float(obj):
    def _float(f):
        return ast.Const(f)
    # }}}

    # {{{ def _long(obj):
    def _long(l):
        return ast.Const(l)
    # }}}

    # {{{ def _str(obj):
    def _str(s):
        return ast.Const(s)
    # }}}

    # {{{ def Symbol(symbol):
    def Symbol(symbol):
        if symbol == data.true:
            return ast.Const(True)
        
        elif symbol == data.false:
            return ast.Const(False)
        
        elif symbol == data.none:
            return ast.Const(None)

        elif symbol.namespace == "":
            return ast.Name(symbol.name)

        else:
            debug()
            raise CompileError, "can't compile symbol with namespace: %s" % symbol
    # }}}

    # {{{ def Doc(doc):
    def Doc(doc):
        rootopCompiler = rootopCompilers.get(doc.tag)

        if rootopCompiler:
            return rootopCompiler(doc)

        else:
            op = language.getOp(doc.tag)
            if op:
                pycompile = getattr(op, "pycompile", None)
                if pycompile:
                    return doc.applyTo(pycompile, op)
                
            # Assume it's a function operator - call op.func
            return compileFunctionCall(ast.Getattr(compileGetOp(doc.tag), "func"),
                                       list(doc.content()),
                                       dict(doc.properties()))
    # }}}

# }}}

# {{{ def compileFunctionCall(pyfunc, argxs, kwxs, starArg, dstarArg):
def compileFunctionCall(pyfunc, argxs, kwxs, starArg=None, dstarArg=None):
    # `pyfunc` is already an ast, other args need translating
    for a in itools.chain(argxs, kwxs.values()):
        assertResult(a, 'pass')

    if starArg:
        assertResult(starArg, "pass")

    if dstarArg:
        assertResult(dstarArg, "pass")

    argsc = map(topy, argxs)
    keywords = [ast.Keyword(str(n), topy(v)) for n,v in kwxs.items()]

    return ast.CallFunc(pyfunc, argsc + keywords,
                        starArg and topy(starArg),
                        dstarArg and topy (dstarArg))
# }}}

# {{{ rootop compilers

# {{{ def compileGetlmodule(doc):
def compileGetlmodule(doc):
    return ast.Subscript(logixglobal("lmodules"),'OP_APPLY',
                         [topy(doc[0])])
# }}}

# {{{ def compileCall(doc):
def compileCall(doc):
    argxs = list(doc.content())
    kwxs = dict(doc.properties())
    
    if len(argxs) == 0:
        raise CompileError, "no function in function-call operator"

    assertResult(argxs[0], "call")

    starS = data.Symbol("", '*')
    dstarS = data.Symbol("", '**')
    
    star = kwxs.get(starS)
    if star is not None:
        del kwxs[starS]
        
    dstar = kwxs.get(dstarS)
    if dstar is not None:
        del kwxs[dstarS]
    
    return compileFunctionCall(topy(argxs[0]), argxs[1:], kwxs, star, dstar)
# }}}

# {{{ def compileList(doc):
def compileList(doc):
    for x in doc:
        assertResult(x, "use in list")
    return ast.List([topy(x) for x in doc])
# }}}

# {{{ def compileTuple(doc):
def compileTuple(doc):
    for x in doc:
        assertResult(x, "use in tuple")
    return ast.Tuple([topy(x) for x in doc])
# }}}

# {{{ def compileSubscript(doc):
def compileSubscript(doc):
    assertResult(doc[0], "subscript")
    for x in doc[1:]:
        assertResult(x, "subscript with")
    return ast.Subscript(topy(doc[0]), 'OP_APPLY', [topy(x) for x in doc[1:]])
# }}}

# {{{ def compileSlice(doc):
def compileSlice(doc):
    assertResult(doc[0], "slice")
    assertResult(doc[1], "slice from")
    assertResult(doc[2], "slice to")
    assertResult(doc[3], "slice with step")
    return ast.Subscript(topy(doc[0]), 'OP_APPLY',
                         [ast.Sliceobj([topy(x) for x in doc[1:4]])])
# }}}

# {{{ def compileDict(doc):
def compileDict(doc):
    for x in doc:
        assertResult(x[0], "usa as dict key")
        assertResult(x[1], "usa as dict value")
    return ast.Dict([(topy(x[0]), topy(x[1])) for x in doc])
# }}}

# {{{ compile getRuleClass
def compileGetRuleClass(doc):
    return ast.Getattr(logixglobal('parser'), doc[0])
# }}}

# {{{ compile setlang:
def compileSetlang(doc):
    langx = doc[0]
    assertResult(langx, "set language to")
    if doc.get('temp'):
        # a top-level setlang - set to a temporary language
        res = ast.CallFunc(logixglobal('tmpLanguage'),
                           [topy(langx), GlobalName('__name__'),
                            ast.CallFunc(GlobalName("globals"), [])])
    else:
        res = topy(langx)

    if hasattr(doc, 'lineno'):
        res.lineno = doc.lineno
    return res
# }}} 

# {{{ compile defop:
def compileDefop(doc):
    binding = doc['binding']
    ruledef = doc['ruledef']
    smartspace = doc.get('smartspace')
    assoc = doc.get('assoc', 'left')
    imp = doc.get('imp')
    lang = doc.get('lang')

    if lang is None:
        raise CompileError("invalid defop")

    assertResult(ruledef, "use in defop")
    assertResult(binding, "use in defop")

    lineno = getattr(doc, "lineno", None)

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
            impkind = 'macro'
            
        elif imp['kind'] == 'f':
            impkind = 'func'

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

        impArgs = [ast.Const(impkind), ast.Name(impfuncname)]
    else:
        impArgs = []

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

    # If unmarshallable objects get into the code-object, you get a
    # blanket error message, so check these before they go in.
    assert isinstance(token, str)
    assert isinstance(smartspace, str) or smartspace == None
    assert isinstance(assoc, str)

    newOpCall = ast.CallFunc(
        ast.Getattr(ast.Getattr(langexpr, '__impl__'), 'newOp'),
        [ast.Const(token),
         topy(binding),
         topy(ruledef),
         ast.Const(smartspace),
         ast.Const(assoc)]
        + impArgs)

    if impArgs != []:
        return ast.Stmt([impfunc, newOpCall])
    else:
        return newOpCall
# }}}

# {{{ compile getops:
def compileGetops(doc):
    fromlang = doc[0]
    targetlang = doc.get("lang")
    ops = [op.strip() for op in doc[1:]]
    
    if targetlang is None:
        raise CompileError("No target language for getops"
                           "(getops in non-exec parse?)")

    assertResult(fromlang, "get operators from")

    lineno = getattr(doc, 'lineno', None)

    lastdot = targetlang.rfind('.')
    if lastdot == -1:
        def targetlangexpr():
            return GlobalName(targetlang)
    else:
        assert 0, ("PROBE: Why are we getting ops into an imported language?")
        langname = targetlang[lastdot+1:]
        langmod = targetlang[:lastdot]
        def targetlangexpr():
            return ast.Getattr(ast.Subscript(logixglobal('lmodules'),
                                             'OP_APPLY',
                                             [ast.Const(langmod)]),
                               langname)

    if len(ops) == 0:
        # get all operators
        res = ast.CallFunc(ast.Getattr(ast.Getattr(targetlangexpr(), '__impl__'),
                                       'addAllOperators'),
                           [topy(fromlang)])
        
    else:
        stmts = [ast.CallFunc(ast.Getattr(ast.Getattr(targetlangexpr(), '__impl__'),
                                          'addOp'),
                              [ast.CallFunc(ast.Getattr(ast.Getattr(topy(fromlang),
                                                                    "__impl__"),
                                                        "getOp"),
                                            [ast.Const(op), ast.Const(False)])])
                 for op in ops]

        for s in stmts: s.lineno = lineno
        res = ast.Stmt(stmts)

    res.lineno = lineno
    return res
# }}}

# {{{ compile deflang:
def compileDeflang(doc):
    name = doc[0]
    parentx = doc[1]
    body = doc["body"]
    
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
        ast.Assign([compilePlace(name)],
                   ast.CallFunc(logixglobal('defLanguage'),
                                [ast.Const(str(name)),
                                 parent,
                                 ast.CallFunc(GlobalName("globals"), [])])),
        astFunction(funcname, tuple(), tuple(), 0, None, funcbody),
        ast.CallFunc(ast.Name(funcname), []),
        ast.AssName(funcname, 'OP_DELETE')])
    
    lineno = getattr(doc, 'lineno', None)
    for n in res.nodes:
        n.lineno = lineno
    res.lineno = lineno
    return res
# }}}

# {{{ compile quote
def compileQuote(doc):
    return quote(doc[0])
# }}}

# {{{ compile escape
def compileEscape(doc):
    raise CompileError("quote-escape outside quote")
# }}}

# {{{ def installRootopCompilers():
def installRootopCompilers():
    rootopCompilers[rootops.callOp] = compileCall
    rootopCompilers[rootops.listOp] = compileList
    rootopCompilers[rootops.tupleOp] = compileTuple
    rootopCompilers[rootops.dictOp] = compileDict
    rootopCompilers[rootops.getlmoduleOp] = compileGetlmodule
    rootopCompilers[rootops.subscriptOp] = compileSubscript
    rootopCompilers[rootops.sliceOp] = compileSlice
    
    rootopCompilers[rootops.getRuleClass] = compileGetRuleClass
    rootopCompilers[rootops.setlang] = compileSetlang
    rootopCompilers[rootops.defop] = compileDefop
    rootopCompilers[rootops.getops] = compileGetops
    rootopCompilers[rootops.deflang] = compileDeflang
    rootopCompilers[rootops.quote] = compileQuote
    rootopCompilers[rootops.escape] = compileEscape
# }}}

installRootopCompilers()
# }}}

# {{{ class OpFuncs:
class OpFuncs:

    # {{{ def continuationOp(self, expr, parts):
    def continuationOp(self, expr, *parts):
        "" # The language knows it as the blank operator
        assertResult(expr, "call or subscript")

        res = topy(expr)

        for part in parts:
            kind = part[-1]
            args = part[:-1]
            if kind == 'call':
                funcArgs = []
                funcKws = {}
                for a in args:
                    if isPyOp(a, "="):
                        if isinstance(a[0], Symbol) and a[0].namespace == "":
                            funcKws[a[0].name] = a[1]
                        else:
                            raise CompileError("invalid keyword arg %r" % a[0])
                    else:
                        funcArgs.append(a)

                star = part.get('star')
                dstar = part.get('dstar')
                
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
            if not isinstance(n, Symbol) or n.namespace != "":
                raise CompileError("invalid global: %s" % n)
        return ast.Global([n.name for n in names])
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
    def _if(self, tests, else_=None):
        for test in tests:
                assertResult(test['test'], "use as if test")
                
        tests = [(el['test'], el['body']) for el in tests]

        pytests = []
        for t,b in tests:
            pyt = topy(t)
            pyt.lineno = getattr(t, 'lineno', None)
            pytests.append( (pyt, block(b, True)) )

        # HACK: For line numbers. Workaround for a test with no metadata
        # (e.g. just a symbol)
        # TODO: make this work for elifs (it doesn't)
        if pytests[0][0].lineno is None:
            pytests[0][0].lineno =  getattr(self, 'lineno', None)

        if else_: else_ = block(else_, True)

        return ast.If(pytests, else_ or ast.Const(None))
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

        if len(body) > 0:
            bod = block(body, True)
            stmts = bod.nodes
            if len(stmts) > 1 and isinstance(stmts[0], ast.Pass):
                first = 1
            else:
                first = 0
            stmt1 = stmts[first]
            if isinstance(stmt1, ast.Const) and isinstance(stmt1.value, str):
                doc = stmt1.value
                del stmts[:first+1]
                if len(stmts) == 0:
                    bod = ast.Const(None)
            else:
                doc = None
        else:
            bod = ast.Const(None)
            doc = None

        return astFunction(str(name), argnames, defaults, flags, doc, ast.Return(bod))
    # }}}

    # {{{ def _lambda(self, args, body):
    def _lambda(self, body, args=None):
        flags, argnames, defaults = funcArgs(args)

        return_ = Symbol(rootops.base_ns, "return")
        do = Symbol(rootops.base_ns, "do:")

        if not hasResult(body):
            body = Doc(do, [body, Doc(return_, [None])])

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

            return ast.List(map(topy, elems))

        def comp(elem, quals):
            assertResult(elem, 'use in list comprehension')

            def qual(place, list, ifs=None):
                ifs = ifs or []
                assertResult(list, "generate list from")
                for test in ifs:
                    assertResult(test, "filter list with")
                return ast.ListCompFor(compilePlace(place),
                                       topy(list),
                                       [ast.ListCompIf(topy(test))
                                        for test in ifs])

            return ast.ListComp(topy(elem), [q.applyTo(qual)
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

        if items == (data.none,):
            return ast.Dict([])
        else:
            for x in items[1::2]: assertResult(x, "use in dict literal")

            return ast.Dict([(topy(items[i]), topy(items[i+1]))
                             for i in range(0, len(items), 2)])
    # }}}

    # {{{ assign = 
    def assign(self, place, val):
        "="
        if isDoc(val, self.symbol):
            # chained assign (a = b = blah)
            res = topy(val)
            res.nodes.append(compilePlace(place))
            return res
        else:
            assertResult(val, "assign from")
            res = ast.Assign([compilePlace(place)], topy(val))
            if lineno:
                res.lineno = lineno
            return res
    # }}}

    # {{{ def _print(self, dest, *args, **kws):
    def _print(self, dest, *args, **kws):
        node = kws.get('nonl') and ast.Print or ast.Printnl

        for arg in args: assertResult(arg, "print")

        asts = [topy(x) for x in args if x != data.none]
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
            return ast.Tuple()
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
        return ast.Import([('.'.join([m.name for m in module]), as and str(as))])
    # }}}

    # {{{ def _from(self, module, imp):
    def _from(self, module, imp):
        if imp == 'star':
            imps = [('*', None)]
        else:
            def tostr(a):
                if a == data.none: return None
                else: return str(a)
            imps = [(tostr(a), tostr(b)) for a,b in zip(imp[::2], imp[1::2])]

        return ast.From(str('.'.join([m.name for m in module])), imps)
    # }}}

    # {{{ def limport(self, module, as):
    def limport(self, module, as=None):
        if as != None:
            if not isinstance(as, Symbol):
                raise CompileError("invalid 'as' clause in input")
        else:
            as = module[-1]
        fname = '.'.join([m.name for m in module])
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

    # {{{ def comma(self, *elems):
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
                                 [e.applyTo(one) for e in excepts],
                                 _else)

        else:
            return ast.TryFinally(block(body, False),
                                  block(kw['finally'], False))
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

# {{{ def compilePlace(place, opname='OP_ASSIGN'):
def compilePlace(place, opname='OP_ASSIGN'):

    # {{{ def compileSubscriptPlace(objx, keyxs):
    def compileSubscriptPlace(objx, keyxs):
         assertResult(objx, "subscript")

         for key in keyxs:
             assertResult(key, "subscript with")

         return ast.Subscript(topy(objx), opname, map(topy, keyxs))
    # }}}

    # {{{ def compileSlicePlace(objx, fromx, tox, stepx):
    def compileSlicePlace(objx, fromx, tox, stepx):
         assertResult(objx, "slice")
         
         assertResult(fromx, "slice from ")
         assertResult(tox, "slice to")
         assertResult(stepx, "slice with step")
         
         return ast.Subscript(topy(objx), opname,
                              [ast.Sliceobj(map(topy, (fromx, tox, stepx)))])
    # }}}
    
    if isinstance(place, Symbol):
        if place.namespace != "":
            raise CompileError, "cannot assign to symbol with namespace: %s" % place
        return ast.AssName(place.name, opname)

    elif isDoc(place, rootops.sliceOp):
        return compileSlicePlace(place[0], place[1], place[2], place[3])

    elif isDoc(place, rootops.subscriptOp):
        return compileSubscriptPlace(place[0], place[1:])

    elif isPyOp(place):
        token = place.tag.name

        if token == ".":
            # {{{ assign to field
            expr = topy(place[0])
            field = place[1]
            if isinstance(field, Symbol) and field.namespace == "":
                return ast.AssAttr(expr, field.name, opname)
            else:
                raise CompileError("Cannot assign to %s" % place)
            # }}}

        elif token == "": # continuation op
            # {{{ assign to slice or subscript
            last = place[-1]

            # expr = the continuationOp not including the last part
            if len(place) == 2:
                expr = place[0]
            else:
                expr = Doc(Symbol(rootops.base_ns, ""),
                           place[:-1])
            
            kind = last[-1]
            if kind == 'subscript':
                return compileSubscriptPlace(expr, last[:-1])

            elif kind ==  "slice":
                start,end,step = last[:-1]
                return compileSlicePlace(expr, start, end, step)
                
            else:
                raise CompileError("Cannot asign to %s " % place)
            # }}}

        elif token == ",":
            return ast.AssTuple([compilePlace(p, opname) for p in list(place)])

        elif token == "[" and place[1] == 'list':
            return ast.AssList([compilePlace(p, opname) for p in place])

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
            dis.lineno = getattr(x, 'lineno', None)
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
            p.lineno = getattr(last, "lineno", None)
        
            statements = stms + [p, lastpy]
        else:
            statements = stms + [lastpy, ast.Const(None)]
    else:
        statements = [stm(s) for s in lst]

    # HACK: No lineno info was emitted for "del a[0]" statements
    # (insert a Pass to fix that)
    s2 = []
    for s in statements:
        if isinstance(s, ast.Subscript) and s.flags == "OP_DELETE" and hasattr(s, "lineno"):
            p = ast.Pass()
            p.lineno = s.lineno
            s2.append(p)
        s2.append(s)

    if prepend:
        return ast.Stmt(prepend + s2)
    else:
        return ast.Stmt(s2)
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

    lineno = None

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
        stmts = [ast.Assign([ast.AssName('__reload__', 'OP_ASSIGN')], ast.List(())),
                 ast.For(ast.AssName('[--codeobj--]', 'OP_ASSIGN'),
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

def compileSymbol(sym):
    return ast.CallFunc(logixglobal('Symbol'), [ast.Const(sym.namespace),
                                                ast.Const(sym.name)])

def localModuleQuote():
    return Doc(rootops.quote,
               [Doc(rootops.getlmoduleOp,
                    [Doc(rootops.escape,
                         [Symbol("", "__name__")],
                         {"extra":[]})])])

# {{{ def quotedDoc(doc, depth):
def quotedDoc(doc, depth):
    """
    Generate code to build a Doc like `doc` with quote-escapes
    replaced by runtime values.
    """

    # Each element of contentParts is either:
    #  an AST - meaning a spliced value - the resulting content
    #           should be `extend`ed by that value
    #  a list of ASTs - a list of regular elements, the resulting
    #                   content should be `extend`ed by an ast.List
    #                   with these elements
    # {{{ contentParts = 
    contentParts = [[]]

    for x in doc.content():
        if isDoc(x, rootops.escape):
            escapelevel = 1 + len(x.get('extra', []))
            if escapelevel > depth:
                raise CompileError("more quote escapes than quotes")
            escape = escapelevel == depth
        else:
            escape = False
            
        if escape:
            if x.hasProperty("splice"):
                assertResult(x[0], "insert into quote")
                contentParts.append(topy(x[0]))
                contentParts.append([])
            elif x.hasProperty("localmodule"):
                contentParts[-1].append(topy(localModuleQuote()))
            else:
                assertResult(x[0], "insert into quote")
                contentParts[-1].append(topy(x[0]))
            
        else:
            contentParts[-1].append(quote(x, depth))
    # }}}

    # These properties are added to the result doc *after*
    # any spliced in docs, so they overwrite any spliced properties
    for v in doc.propertyValues():
        assertResult(v, "use as operand")
    properties = ast.Dict([(compileSymbol(n), quote(v, depth))
                           for n, v in doc.properties()])
        
    # assert isinstance(contentParts[0], list)

    if contentParts == [[]]:
        return ast.CallFunc(logixglobal("Doc"),
                            [compileSymbol(doc.tag),
                             properties])
    else:
        if len(contentParts[0]) > 0:
            docArg = ast.List(contentParts[0])
            rest = contentParts[1:]
        elif len(contentParts) > 0:
            docArg = contentParts[1]
            rest = contentParts[2:]
        
        if len(rest) == 0:
            return ast.CallFunc(logixglobal("Doc"),
                                [compileSymbol(doc.tag),
                                 docArg,
                                 properties])
        else:
            val = macros.gensym("val")
            stmts = [ast.Assign([compilePlace(val)],
                                ast.CallFunc(logixglobal("Doc"),
                                             [compileSymbol(doc.tag),
                                              docArg]))]
            for part in rest:
                if isinstance(part, list):
                    if len(part) == 0:
                        continue
                    ext = ast.List(part)
                else:
                    ext = part
                stmts.append(ast.Discard(ast.CallFunc(ast.Getattr(topy(val),
                                                                  "extend"),
                                                      [ext])))
        
            stmts.append(ast.Discard(ast.CallFunc(ast.Getattr(topy(val),
                                                              "extend"),
                                                  [properties])))
            stmts.append(topy(val))
            return ast.Stmt(stmts)
# }}}

def quote(obj, depth=1):
    assert depth > 0

    if isDoc(obj, rootops.quote):
        depth += 1

    elif isDoc(obj, rootops.escape):
        escapelevel = 1 + len(obj.get('extra', []))
        if escapelevel > depth:
            raise CompileError("more quote escapes than quotes")
        
        elif escapelevel == depth:
            if obj.hasProperty('splice'):
                raise CompileError("Can't splice here")

            elif obj.hasProperty("localmodule"):
                return topy(localModuleQuote())
            
            else:
                assertResult(obj[0], "insert into quote")
                return topy(obj[0])

    if isinstance(obj, Doc):
        return quotedDoc(obj, depth)

    elif isinstance(obj, Symbol):
        return compileSymbol(obj)

    else:
        return topy(obj)
# }}}
