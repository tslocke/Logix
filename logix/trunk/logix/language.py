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
import types
import sys
import inspect

import data
from data import *

try:
    from livedesk.util import debug, debugmode
except ImportError: pass
# }}}

# {{{ class Operator(object):
class Operator(object):

    def __init__(self, syntax, language):
        self.syntax = syntax
        self.language = language
        self.symbol = Symbol(language.__impl__.name, syntax.token)

    def __getattr__(self, name):
        return getattr(self.syntax, name)

    def expandMacro(self, doc, context=None):
        args = list(doc.content())
        kwargs = {}
        for name, val in doc.properties():
            kwargs[str(name)] = val
        
        if "__context__" in inspect.getargspec(self.macro)[0]:
            kwargs["__context__"] = context

        return self.macro(*args, **kwargs)

    def __repr__(self):
        return "<Operator %s>" % self.symbol
# }}}

# {{{ class Language:
class Language(object):

    allLanguages = weakref.WeakValueDictionary()

    def __init__(self, name, parent=None, module=None):
        self.__impl__ = LanguageImpl(self, name, parent)
        self.__module__ = module
        Language.allLanguages[name] = self

    # {{{ def __repr__(self):
    def __repr__(self):
        return "<language %s @%x>" % (self.__impl__.name, id(self))
    # }}}

    # {{{ def __add__(self, other):
    def __add__(self, other):
        if not isinstance(other, Language):
            raise TypeError, "can only add a Language (not a $(type other)) to a Language"

        new = Language("%s+%s" % (self.__impl__.name, other.__impl__.name), self)
        new.__impl__.addAllOperators(other)
        return new
    # }}}

# }}}

# {{{ class LanguageImpl(object):
class LanguageImpl(object):
    
    # {{{ vars
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
        self.invalidateRegexes()
        self.clearBlockCache()
    # }}}

    # {{{ def derivedLanguages(self):
    def derivedLanguages(self):
        return self._derivedlangs.keys()
    # }}}

    # {{{ def addOp(self, op):
    def addOp(self, op):
        token = op.token
        self.operators[token] = op
        
        if op != self.getContinuationOp():
            self.addToken(token)
            
        rrule = op.rightRule
        if rrule:
            for t in rrule.allLiterals():
                self.addToken(t)
        self.invalidateRegexes()
        self.clearBlockCache()

        return op
    # }}}

    # {{{ def getOp(self, token, quiet=True):
    def getOp(self, token, quiet=True):
        op = self.operators.get(token)

        if op is None:
            for p in self.parents:
                op = p.operators.get(token)
                if op is not None: return op

        if op is False:
            op = None

        if op != None or quiet:
            return op
        else:
            raise ValueError, "no such operator %s:%s" % (self.name, token)
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

        if mode in ("exec", "execmodule", "interactive"):
            if execenv is None:
                raise ValueError, "you must pass an execenv with parse mode %r" % mode
        elif mode in ("parse", "expand"):
            if execenv is not None:
                raise ValueError, "you must not pass an execenv with parse mode %r" % mode
        else:
            raise ValueError, "invalid parse mode %r" % mode
                
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
            parselang = tmpLanguage(self.userlang, execenv['__name__'], execenv)

            # Important - retrieve the lang back from the env
            # The UpdatingEnv may have merged it into an existing object
            parselang = execenv[parselang.__impl__.name].__impl__
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
                assert exc != None
            else:
                token = tokenizer.nextToken()
                if token == parser.failed or not tokenizer.atEOF():
                    exc = tokenizer.furthestError
                    assert exc != None
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
                execenv['__noreload__'] = []
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
                    return macros.expand(lines)
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

    # {{{ def addToken(self, token):
    def addToken(self, token):
        def isName(s):
            m = parser.Tokenizer.nameRx.match(s)
            return m is not None and m.end() == len(s)
        
        if not isName(token) and token not in self.toks:
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
    
    # {{{ def invalidateRegexes(self):
    def invalidateRegexes(self):
        self.regex = None
            
        for lang in self.derivedLanguages():
            lang.invalidateRegexes()
    # }}}

    # {{{ def regexValid(self):
    def regexValid(self):
        parent = self.parent
        return self.regex != None and (parent == None or parent.regexValid())
    # }}}
    # }}}

    # {{{ def newOp(self, token, binding, ruledef, smartspace=None, assoc='left',
    def newOp(self, token, binding, ruledef, smartspace=None, assoc='left',
              impKind=None, impFunc=None):
        if token == "__continue__":
            syntax = parser.OperatorSyntax("", binding, ruledef,
                                           "right", None)
            syntax.leftRule = parser.ExpressionRule()
        else:
            syntax = parser.OperatorSyntax(token, binding, ruledef,
                                           assoc, smartspace)
        return self.newOpFromSyntax(syntax, impKind, impFunc)
    # }}}
        
    # {{{ def newOpFromSyntax(self, syntax, impKind=None, impFunc=None):
    def newOpFromSyntax(self, syntax, impKind=None, impFunc=None):
        op = Operator(syntax, self.userlang)
        if syntax.token == "":
            self.continuationOp = op

        if impKind == 'macro':
            op.macro = impFunc
        else:
            op.func = impFunc

        self.addOp(op)
        return op
    # }}}

    # {{{ def addDeflangLocals(self, attrs):
    def addDeflangLocals(self, attrs):
        for name, val in attrs.items():
            if not name.startswith("_"):
                setattr(self.userlang, name, val)
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

    # {{{ def getAllOperators(self):
    def getAllOperators(self):
        mine = set(self.operators.values())
        if self.parent:
            return mine | self.parent.getAllOperators()
        else:
            return mine
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
        copy = [data.copy(line) for line in lines]
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

        for l in self.derivedLanguages():
            l.clearBlockCache()
    # }}}

    def clearAllCaches():
        for lang in Language.allLanguages.values():
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
    # Important - retrieve the lang back from the env
    # The UpdatingEnv may have merged it into an existing object
    if globals:
        return globals[name]
    else:
        return l
# }}}

# {{{ def cachedCodeCopy(code, linedelta):
def cachedCodeCopy(codeDoc, linedelta=None):
    res = data.copy(codeDoc)

    def applyLineDelta(x):
        if hasattr(x, "lineno"):
            x.lineno += linedelta
        if isinstance(x, Doc):
            for subx in x.propertyValues():
                applyLineDelta(subx)
            for subx in x.content():
                applyLineDelta(subx)

    applyLineDelta(res)
    return res
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
        assert 0, "PROBE: didn't expect to get here"
        globs = sys._getframe(-1).f_globals
    existing = globs.get(name)
    if isinstance(existing, Language):
        if ((parent is None and existing.__impl__.parent is not None)
            or (parent is not None
                and (existing.__impl__.parent is None
                     or existing.__impl__.parent.userlang is not parent))):
            raise LanguageBaseException("base language for %s differs from"
                                        " forward declaration" % name)
        
        existing.__impl__.clear()
        # Q: Is it ok if \existing is imported from another module?
        return existing
    else:
        return Language(name, parent, globs['__name__'])
# }}}

# {{{ def getLanguage(namespace):
def getLanguage(namespace):
    return Language.allLanguages.get(namespace)
# }}}

# {{{ def getOp(symbol):
def getOp(symbol):
    lang = getLanguage(symbol.namespace)
    return lang and lang.__impl__.getOp(symbol.name)
# }}}

import parser, macros
import pycompile as compiler
import rootops
