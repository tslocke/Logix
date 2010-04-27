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
import weakref
import StringIO
import re
import new
import types
import sys
import inspect

from data import *

try:
    from livedesk.util import debug, debugmode
except ImportError: pass
# }}}

# {{{ def instantiateOp(clazz, operands, lineno):
def instantiateOp(opclass, operands, lineno):
    import rootops
    cons = getattr(opclass, "__init__", None)
    if cons and cons is not object.__init__:
        op = opclass(*operands.elems, **operands.fields)
    else:
        op = opclass()
        op.__operands__ = operands

    if lineno:
        setmeta(op, lineno=lineno)
    return op
# }}}

# {{{ def isOperatorType(t):
def isOperatorType(t):
    if issubclass(t, Operator):
        return True
    else:
        return hasattr(t, '__syntax__')
# }}}

# {{{ class OperatorType(type):
class OperatorType(type):

    def __repr__(self):
        if hasattr(self, '__language__') and hasattr(self, '__syntax__'):
            return '<operator %s %s>' % (self.__language__.__impl__.name,
                                         self.__syntax__.token)
        else:
            return type.__repr__(self)
# }}}

# {{{ class Operator(object):
class Operator(object):

    __metaclass__ = OperatorType

    def __pprint__(self, indent, parens=False):
        cls = self.__class__
        syn = getattr(cls, '__syntax__', None)
        if syn:
            head = cls.__language__.__impl__.name + ":" + syn.token
        else:
            head = cls.__name__

        if parens:
            lparen, rparen = "()"
        else:
            lparen = rparen = ""
        return pprint(head, opargs(self), self.__operands__.items(), indent,
                      lparen=lparen, rparen=rparen)

    def __repr__(self):
        return "<%s>" % self.__pprint__(1)
# }}}

# {{{ class BaseOperator(Operator):
class BaseOperator(Operator):

    def __init__(self, *args, **kws):
        self.__operands__ = flist.new(args, kws)

    def __getitem__(self, key):
        return self.__operands__[key]

    def __setitem__(self, key, value):
        self.__operands__[key] = value

    def __len__(self):
        return len(self.__operands__)

    def __iter__(self):
        return self.__operands__.__iter__()
# }}}

# {{{ class BaseMacro(BaseOperator):
class BaseMacro(BaseOperator):

    def __macro__(self, context=None):
        mfunc = self.__class__.macro
        kws = opfields(self)
        if "__context__" in inspect.getargspec(mfunc)[0]:
            kws["__context__"] = context

        return mfunc(*opargs(self), **kws)
# }}}

# {{{ class Language:
class Language(object):

    allLanguages = weakref.WeakKeyDictionary()

    operatorBase = BaseOperator

    def __init__(self, name, parent=None, module=None):
        self.__impl__ = LanguageImpl(self, name, parent)
        self.__module__ = module
        Language.allLanguages[self] = True

    # {{{ def __repr__(self):
    def __repr__(self):
        return "<language %s @%x>" % (self.__impl__.name, id(self))
    # }}}

    # {{{ def __redef__(self, other, globs, updater):
    def __redef__(self, other, globs, updater):
        from livedesk.util import newClassLike

        if len(vars(other)) == 2:
            # Looks like a forward declaration
            return
        
        l = self.__impl__
        ops = l.operators
        otherops = other.__impl__.operators
        for tok, op in ops.items():
            otherop = otherops[tok]
            if tok in otherops:
                op.__syntax__ = otherop.__syntax__
                updater(op, otherop)
            else:
                del l.operators[tok]

        for tok,clss in otherops.items():
            if tok not in ops:
                print "Add op to %s: %s" % (l.name, tok)
                c = newClassLike(clss, globs)
                updater(c, clss)
                l.addOp(c)

        for name, val in self.__dict__.items():
            if (type(val) == types.FunctionType
                and isinstance(getattr(other, name, None), types.FunctionType)):
                updater(val, getattr(other, name))
            

        l.toks = other.__impl__.toks
        l.tokens = other.__impl__.tokens
        l.invalidateRegexes()
        l.clearBlockCache()
    # }}}    
# }}}

# {{{ class LanguageImpl(object):
class LanguageImpl(object):
    
    # {{{ vars
    configVars = ['defaultOp',
                  'operatorBase']
    continuationOp = None
    maxCacheSize = 1e6
    # }}}

    # {{{ def __init__(self, name, parent=None):
    def __init__(self, userlang, name, parent=None):
        self.userlang = userlang
        self.name = name
        self.parent = parent and parent.__impl__

        if parent:
            self.parent._derivedlangs[self] = 1
        self._derivedlangs = weakref.WeakKeyDictionary()

        # {{{ getOp optimisation
        #
        # language.getOp gets called a *lot* during parsing.
        # Here we assume the ancestor chain won't change, and we make a list of
        # parents. Iterating over this list is much faster than recursively
        # calling parent.getOp()
        
        self.parents = []
        p = self.parent
        while p != None:
            self.parents.append(p)
            p = p.parent
        # }}}

        class CacheDict(dict):
            def __repr__(self):
                return "<block-cache for '%s'>" % name
        self._blockCache = CacheDict()

        self.clear()
    # }}}

    # {{{ def clear(self):
    def clear(self):
        self.operators = {}
        self.tokenMacros = {}
        self.toks = []
        self.tokens = []
        self.optextTokens = []
        self.optextRegexes = {}
        self.invalidateRegexes()
        self.clearBlockCache()
    # }}}

    # {{{ def derivedLanguages(self):
    def derivedLanguages(self):
        return self._derivedlangs.keys()
    # }}}

    # {{{ def addOp(self, op):
    def addOp(self, op):
        def isName(s):
            m = parser.Tokenizer.nameRx.match(s)
            return m is not None and m.end() == len(s)

        #Q: should __language__ be a field of OperatorSyntax instead?
        if getattr(op, "__language__", None) is None:
            op.__language__ = self.userlang

        opsyn = op.__syntax__
        token = opsyn.token
        self.operators[token] = op

        # Any prefix operator is valid in optext
        
        if ((opsyn.leftRule is None and opsyn.binding == 0)
            or opsyn.isEnclosed()):
            self.optextTokens.append( (len(token), re.escape(token)) )

        # Don't add regular names
        if not isName(token):
            self.addToken(token)
            
        rrule = op.__syntax__.rightRule
        if rrule:
            for t in rrule.allLiterals():
                # Don't add regular names
                if not isName(t):
                    self.addToken(t)
        self.invalidateRegexes()
        self.clearBlockCache()

        return op
    # }}}

    # {{{ def getOp(self, token):
    def getOp(self, token):
        op = self.operators.get(token)

        if op is None:
            for p in self.parents:
                op = p.operators.get(token)
                if op is not None: return op

        if op is False:
            return None
        else:
            return op
    # }}}

    # {{{ def hasop(self, token):
    def hasop(self, token):
        return self.operators.has_key(token)
    # }}}

    # {{{ def delOpXXX(self, token):
    def delOpXXX(self, token):
        # We should also corresponding stuff tokens from self.toks and self.toks
        # but we don't because a) we don't have enough info to know what to remove
        # and b) semantics are not effected (only regex performance)
        
        if token in self.operators or not self.getOp(token) != None:
            raise ValueError("no such inherited operator '%s'" % token)
        else:
            self.operators[token] = False
    # }}}

    # {{{ def parse(self, ...):
    def parse(self, src, filename=None, execenv=None, mode='parse'):
        # {{{ DOC
        """ Modes:
            exec = Expand macros, and exec top-level items
                   (must supply an execenv)#
            execmodule = Like exec but returns a codeobject
            interactive = Expand macros, Eval language expressions only
                          (e.g. in switchlang), returns code-data
                          (must supply an execenv)
            parse = Parse only, no macro expansion, no eval, returns code-data
            expand = Parse and expand macros, no eval, returns code-data
                          
        """
        # }}}
        
        assert ((mode in ("exec", "execmodule", "interactive") and execenv is not None)
                or (mode in ("parse", "expand") and execenv is None))
                
        # {{{ input = 
        if type(src) == str:
            input = StringIO.StringIO(src)
        elif (hasattr(src, 'readline')
              and hasattr(src, 'tell')
              and hasattr(src, 'seek')):
            input = src
        else:
            raise parser.ParseError("cannot parse %s" % (src,))
        # }}}

        tokenizer = parser.Tokenizer()

        # {{{ fname = get filename
        if filename:
            fname = filename
        else:
            name = getattr(src, 'name', None)
            if name:
                fname = name
            elif type(src) == str:
                fname = '<string>'
            else:
                fname = '<unknown>'
        # }}}

        if mode == 'execmodule':
            parselang = tmpLanguage(self.userlang, execenv['__name__']).__impl__
            execenv[parselang.name] = parselang.userlang
        else:
            parselang = self
            
        tokenizer.setInput(input, fname)
        tokenizer.startLanguage(parselang)

        # {{{ def handleException(res, tokenizer):
        def handleException(res, tokenizer):
            if isinstance(res, parser.ParseError):
                exc = res
            elif res == parser.failed:
                exc = tokenizer.furthestError
            else:
                token = tokenizer.nextToken()
                if token == parser.failed:
                    exc = tokenizer.furthestError
                elif not tokenizer.atEOF():
                    exc = parser.ParseError('syntax error', *tokenizer.getPos())
                else:
                    exc = None

            if exc:
                if type(src) == str:
                    lines = src.splitlines()
                elif type(src) == file:
                    src.seek(0)
                    lines = src.readlines()

                if exc.lineno and exc.lineno <= len(lines):
                    line = lines[exc.lineno-1]
                    lineno = exc.lineno
                else:
                    line = ""
                    lineno = 1
                raise SyntaxError, (str(exc), (fname, lineno, exc.offset, line))
        # }}}

        try:
            if mode in ('exec', 'execmodule'):
                # {{{ parse TopLevelBlock, return code-data list or codeobject
                lines, codeobjects = parser.TopLevelBlockRule().parse(
                    tokenizer, tokenizer.finalTerminator, execenv)
                handleException(lines, tokenizer)
                if mode == 'execmodule':
                    return compiler.compileCodeObjects(fname, codeobjects)
                else:
                    return lines
                # }}}

            elif mode == "interactive":
                # {{{ parse ExpressionRule, return code-data (macro expanded)
                expr = parser.ExpressionRule().parse(
                    tokenizer, tokenizer.finalTerminator, execenv)
                handleException(expr, tokenizer)
                return macros.expand(expr)
                # }}}

            else: # assert mode in ("parse", "expand")
                # {{{ parse BlockRule return code-data (maybe macro expanded)
                lines = parser.BlockRule().parse(
                    tokenizer, tokenizer.finalTerminator, execenv)
                handleException(lines, tokenizer)
                if mode == 'expand':
                    return map(macros.expand, lines)
                else:
                    return lines
                # }}}

        except parser.ParseError, e:
            handleException(e, tokenizer)
    # }}}

    # {{{ def getContinuationOp(self):
    def getContinuationOp(self):
        return self.continuationOp or (self.parent
                                       and self.parent.getContinuationOp())
    # }}}
    
    # {{{ def setTokenMacro(self, kind, macrofunc):
    def setTokenMacro(self, kind, macrofunc):
        self.tokenMacros[kind] = macrofunc
        self.clearBlockCache()
    # }}}

    # {{{ def getTokenMacro(self, kind):
    def getTokenMacro(self, kind):
        p = self.parent
        return self.tokenMacros.get(kind, p and p.getTokenMacro(kind))
    # }}}

    # {{{ Regexes
    # {{{ def rebuildRegex(self):
    def rebuildRegex(self):
        tokens = self.getTokens()
        rx = '|'.join([c[1] for c in tokens])
        self.regex = re.compile(rx)
    # }}}

    # {{{ def matchToken(self, s, pos):
    def matchToken(self, s, pos):
        if not self.regexValid():
            self.rebuildRegex()
        m = self.regex.match(s, pos)
        if m and m.start() == m.end():
            return None
        else:
            return m
    # }}}

    # {{{ def optextRegex(self, terminator):
    def optextRegex(self, terminator):
        rx = self.optextRegexes.setdefault(terminator, None)
            
        if rx is None:
            optextTokens = self.getOptextTokens()

            # Note: We put the terminator at the start of the regex, so optext
            # cannot include operators that have the terminator as a prefix

            # The terminator is in a group, operators are not
            # This is used by parser.Tokenizer.optextUntil
            rxtext = '|'.join(["(%s)" % terminator]
                              + ["%s" % c[1] for c in optextTokens])
            
            rx = re.compile(rxtext)
            self.optextRegexes[terminator] = rx

        return rx
    # }}}

    # {{{ def addToken(self, token):
    def addToken(self, token):
        if token not in self.toks:
            self.tokens.append( (len(token), re.escape(token)) )
            self.toks.append(token)
    # }}}

    # {{{ def getTokens(self):
    def getTokens(self):
        ptokens = self.parent and self.parent.getTokens() or []
        res = ptokens + self.tokens
        res.sort()
        res.reverse()
        return res
    # }}}
    
    # {{{ def getOptextTokens(self):
    def getOptextTokens(self):
        ptokens = self.parent and self.parent.getOptextTokens() or []
        res = ptokens + self.optextTokens
        res.sort()
        res.reverse()
        return res
    # }}}

    # {{{ def invalidateRegexes(self):
    def invalidateRegexes(self):
        self.regex = None
        for termintor in self.optextRegexes:
            self.optextRegexes[termintor] = None
            
        for lang in self.derivedLanguages():
            lang.invalidateRegexes()
    # }}}

    # {{{ def regexValid(self):
    def regexValid(self):
        parent = self.parent
        return self.regex != None and (parent == None or parent.regexValid())
    # }}}
    # }}}

    # {{{ def getOperatorBase(self):
    def getOperatorBase(self):
        return self.userlang.operatorBase or \
               (self.parent and self.parent.getOperatorBase())
    # }}}

    # {{{ def newOp(self, token, binding, ...):
    def newOp(self, token, binding, ruledef,smartspace=None, assoc='left',
              impKind=None, impFunc=None):
        syntax = parser.OperatorSyntax(token, binding, ruledef,
                                       assoc, smartspace)

        attrs = {
            '__syntax__': syntax,
            }

        base = self.getOperatorBase()
        if '__metaclass__' not in vars(base):
            attrs['__metaclass__'] = OperatorType


        if impKind == 'macro':
            attrs['__macro__'] = BaseMacro.__macro__.im_func

        if impFunc:
            attrs[impKind] = staticmethod(impFunc)

        clss = new.classobj("Operator[%s.%s]" % (self.name, token),
                            (base,),
                            attrs)

        if token == "__continue__":
            clss.__syntax__.leftRule = parser.ExpressionRule()
            clss.__syntax__.assoc = 'right'
            clss.__syntax__.smartspace = None
            clss.__language__ = self.userlang
            
            def isName(s):
                m = parser.Tokenizer.nameRx.match(s)
                return m is not None and m.end() == len(s)
            
            for t in clss.__syntax__.rightRule.allLiterals():
                # Don't add regular names
                if not isName(t):
                    self.addToken(t)

            def __pprint__(self, indent, parens=False):
                if parens:
                    lparen, rparen = "()"
                else:
                    lparen = rparen = ""
                return pprint(self.__language__.__impl__.name + ":",
                              opargs(self), self.__operands__.items(), indent,
                              lparen=lparen, rparen=rparen)
            clss.__pprint__ = __pprint__

            self.continuationOp = clss

        self.addOp(clss)
    # }}}

    # {{{ def addDeflangLocals(self, attrs):
    def addDeflangLocals(self, attrs):
        for name, val in attrs.items():
            if not name.startswith("_"):
                setattr(self.userlang, name, val)

        #conOp = getattr(self, 'continuationOp', None)
        #if conOp:
        #    conOp.__language__ = self.userlang
        #    conOp.__syntax__.assoc = 'right'
        #    conOp.__syntax__.smartspace = None
        #    def isName(s):
        #        m = parser.Tokenizer.nameRx.match(s)
        #        return m is not None and m.end() == len(s)
        #    
        #    for t in conOp.__syntax__.rightRule.allLiterals():
        #        # Don't add regular names
        #        if not isName(t):
        #            self.addToken(t)
    # }}}

    # {{{ def isSublangOf(self, lang):
    def isSublangOf(self, lang):
        return lang is self or lang in self.parents
    # }}}

    # {{{ def addAllOperators(self, fromlang):
    def addAllOperators(self, fromlang):
        ifromlang = fromlang.__impl__

        added = []
        l = ifromlang
        while l is not None and not self.isSublangOf(l):
            for token, op in l.operators.items():
                if token != "__continue__":
                    if token in added:
                        # already added overriden operator
                        pass
                    else:
                        self.addOp(op)

            l = l.parent
    # }}}

    # {{{ Cache
    # {{{ def blockCache(self, blocktext, lineno):
    def blockCache(self, blocktext, lineno):
        cached = self._blockCache.get(blocktext)
        if cached:
            lines, oldlineno, textid = cached
            # {{{ move to front of recently-used queue
            lines, code, textid = cached
            for i, id in enumerate(self._blockCacheUsed):
                if id == textid:
                    del self._blockCacheUsed[i]
                    self._blockCacheUsed.append(id)
                    break
            else:
                assert 0, "id not in _blockCacheUsed"
            # }}}

            #print "CACHE HIT! %s:" % self
            #print blocktext[:20], '...', blocktext[-20:]

            return cachedCodeCopy(lines, lineno-oldlineno)
        else:
            return None
    # }}}

    # {{{ def setBlockCache(self, blocktext, lines, lineno):
    def setBlockCache(self, blocktext, lines, lineno):
        #print '---CACHE STORE--- %s:' % self
        #print blocktext.splitlines()[0]
        #print '...'
        #print blocktext.splitlines()[-1]
        #print
        
        textid = id(blocktext)
        copy = [codeCopy(line) for line in lines]
        self._blockCache[blocktext] = (copy, lineno, textid)
        self._blockCacheUsed.append(textid)
        self._blockCacheSize += len(blocktext)

        # {{{ remove old caches if cache is too big
        while self._blockCacheSize > LanguageImpl.maxCacheSize:
            delid = self._blockCacheUsed[0]
            for text, (lines, lineno, textid) in self._blockCache.items():
                if textid == delid:
                    del self._blockCache[text]
                    del self._blockCacheUsed[0]
                    self._blockCacheSize -= len(text)
                    break
            else:
                assert 0, "id not in _blockCache"
        # }}}
    # }}}

    # {{{ def clearBlockCache(self):
    def clearBlockCache(self):
        self._blockCache.clear()
        self._blockCacheSize = 0
        self._blockCacheUsed = []
    # }}}

    def clearAllCaches():
        for lang in Language.allLanguages.keys():
            lang.__impl__.clearBlockCache()
    clearAllCaches = staticmethod(clearAllCaches)
    # }}}

    # {{{ def __repr__(self):
    def __repr__(self):
        return "<language-impl %s @%x>" % (self.name, id(self))
    # }}}
# }}}

# {{{ def tmpLanguage(lang, modname, globals=None):
def tmpLanguage(lang, modname, globals=None):
    name = "%s~" % lang.__impl__.name
    l = Language(name, lang, modname)
    if globals is not None:
        globals[name] = l
    return l
# }}}

# {{{ def cachedCodeCopy(code, linedelta):
def cachedCodeCopy(code, linedelta=None):
    t = type(code)
    
    if t == flist:
        args = [cachedCodeCopy(x, linedelta) for x in code.elems]
        fields  = dict([(name, cachedCodeCopy(val, linedelta))
                        for name, val in code.items()])

        res = flist.new(args, fields)
        copymeta(code, res)
        return res
    
    elif issubclass(t, (tuple, list)):
        return t([cachedCodeCopy(x, linedelta) for x in code])

    elif t == dict:
        return dict([(name, cachedCodeCopy(x, linedelta))
                     for name, x in code.items()])


    elif t == Symbol and getmeta(code, 'lineno') is not None:
        sym = Symbol(code)
        setmeta(sym, lineno=getmeta(code, 'lineno')+linedelta)
        return sym

    elif isOperatorType(t):
        operands = cachedCodeCopy(code.__operands__, linedelta)
        op = instantiateOp(t, operands, None)
        copymeta(code, op)
        lineno = getmeta(code, 'lineno')
        if lineno is not None:
            setmeta(op, lineno=lineno+linedelta)
        return op

    else:
        assert t in (int, float, long, str, bool, Symbol, types.NoneType,
                     OperatorType), debug()
        return code
# }}}

# {{{ def codeCopy(code):
def codeCopy(code):
    t = type(code)
    
    if t in (tuple, list):
        return t([codeCopy(x) for x in code])

    elif t == dict:
        return dict([(name, codeCopy(x)) for name, x in code.items()])

    elif t == flist:
        args = [codeCopy(x) for x in code.elems]
        fields  = dict([(name, codeCopy(x)) for name, x in code.items()])
        res = flist.new(args, fields)
        copymeta(code, res)
        return res

    elif t == Symbol:
        sym = Symbol(code)
        copymeta(code, sym)
        return sym

    elif isOperatorType(t):
        operands = codeCopy(code.__operands__)
        op = instantiateOp(t, operands, None)
        copymeta(code, op)
        return op

    else:
        assert t in (int, float, long, str, bool, Symbol, types.NoneType,
                     OperatorType)
        return code
# }}}

# {{{ def eval(x, globs=None, locs=None):
_sys_eval = eval
def eval(x, globs=None, locs=None):
    if globs is not None and locs is None:
        pass
    if globs is None and locs is None:
        frm = sys._getframe(1)
        locs  = frm.f_locals
        globs = frm.f_globals
        del frm

    import pycompile as compiler
    line = macros.expand(x)
    code = compiler.compile([line], '<string>', 'eval',
                            module=globs.get('__name__'))

    if locs is not None:
        return _sys_eval(code, globs, locs)
    else:
        return _sys_eval(code, globs)
# }}}

# {{{ def defLanguage(name, parent, globs=None)
class LanguageBaseException(Exception):
    pass

def defLanguage(name, parent, globs=None):
    if globs is None:
        import sys
        debug()
        globs = sys._getframe(-1).f_globals
    existing = globs.get(name)
    if isinstance(existing, Language):
        if ((parent is None and existing.__impl__.parent is not None)
            or (parent is not None and
                existing.__impl__.parent.userlang is not parent)):
            raise LanguageBaseException("base language for %s differs from"
                                        " forward declaration" % name)
        
        existing.__impl__.clear()
        # Q: Is it ok if \existing is imported from another module?
        return existing
    else:
        return Language(name, parent, globs['__name__'])
# }}}

import parser, macros
import pycompile as compiler
import rootops
