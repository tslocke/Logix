# {{{ GPL License
# Logix - an extensible programming language for Python
# Copyright (C) 2004 LiveLogix Corp, USA (www.livelogix.com)
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
# along with this program in the file LICENSE.txt; if not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA  02111-1307, USA.
# }}}
# {{{ imports
import inspect
import re
import StringIO
import tokenize
import itertools as itools

from util import record, concat
from data import Symbol, Meta, getmeta, setmeta, flist
from language import Language, LanguageImpl, instantiateOp,\
     LanguageBaseException, defLanguage, OperatorType

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

# {{{ class ParseError:
class ParseError(SyntaxError):

    def __init__(self, message, lineno=None, col=None, language=None):
        assert ((lineno is None or isinstance(lineno, int))
                and (col is None  or isinstance(col, int))
                and isinstance(message, str))
        
        if language:
            self.language = language
            message += " (language=%s)" % language.name

        self.lineno = lineno
        self.col = col
        self.offset = col is not None and col+1
        self.message = message

        SyntaxError.__init__(self, (message, (None, lineno, self.offset)))
        assert isinstance(self.message, str)


    def __str__(self):
        if self.lineno == None:
            return str(self.message)
        else:
            error = '%s: line %s, col %s' % (self.message,
                                             self.lineno,
                                             self.offset)
            return error

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)
# }}}

# {{{ class RuleError:
class RuleError(Exception):
    pass
# }}}

# {{{ failed = 
class Failed:
    def __repr__(self): return "<parser.failed>"
failed = Failed()
# }}}

# {{{ Tokenizer
# {{{ class Token:
class Token:

    def __init__(self, text, val=None):
        assert text != ''
        self.text = text
        self.value = val

    def __eq__(self, other):
        return isinstance(other, Token) and other.text == self.text

    def __ne__(self, other):
        return not other == self

    def __repr__(self):
        return "Token('%s')" % self.text

    def getOp(self, language):
        return language.getOp(self.text)

    def packed(self):
        return not self.leftSpace or not self.rightSpace
    packed = property(packed)
# }}}

# {{{ class LayoutToken:
class LayoutToken:

    def __init__(self, text):
        self.text=text

    def __repr__(self):
        return "<LayoutToken %s>" % self.text
# }}}
        
# {{{ class Tokenizer:
class Tokenizer:

    endOfLine = LayoutToken('EOL')
    endOfBlock = LayoutToken('EOB')
    endOfFile = LayoutToken('EOF')

    NEXT_LINE_BLOCK = 1
    SAME_LINE_BLOCK = 2

    # {{{ regexes
    commentRx = re.compile(tokenize.Comment)
    # {{{ _number = 
    _number =  '((\\d+[jJ]|(\\d+\\.\\d+([eE][-+]?\\d+)?|\\d+[eE][-+]?\\d+)[jJ])|(\\d+\\.\\d+([eE][-+]?\\d+)?|\\d+[eE][-+]?\\d+)|(0[xX][\\da-fA-F]*[lL]?|0[0-7]*[lL]?|[1-9]\\d*[lL]?))'
    # }}}
    stdtokensRx = re.compile('|'.join(['(?P<%s>%s)' % (name, rx)
                                       for (rx, name) in
                                       [(_number, 'number'),
                                        (tokenize.Name, 'symbol')]]))

    whitespaceRx = re.compile(r"[ \f\t\n\r]+")
    nameRx = re.compile(tokenize.Name)
    blanklineRx = re.compile(r"^[ \t]*(?:\#.*)?\r?$")
    indentRx = re.compile(r"^[ \t]*")
    # }}}
    
    # {{{ def __init__(self):
    def __init__(self):
        self.tokenqueue = []
    # }}}

    # {{{ def setInput(self, input, filename):
    def setInput(self, input, filename):
        self.input = input
        self.filename = filename
        self.blocks = []
        self.marks = []
        self.tokenqueue = []
        self.lineno = 0
        self.col = 0
        self.linestart = 0
        self.currentline = ''
        self.furthestError = None
        self.languageStack = []
    # }}}

    # {{{ def nextLine(self, freetext):
    def nextLine(self, freetext):
        eof = False
        lineno = self.lineno
        while 1:
            self.linestart = self.input.tell()
            lineno += 1
            line = self.input.readline().expandtabs()

            if line == '':
                # end of file
                for _ in self.blocks:
                    self.tokenqueue.append( (Tokenizer.endOfBlock,
                                             self.lineno,
                                             self.col) )
                self.tokenqueue.append( (Tokenizer.endOfFile,
                                         self.lineno,
                                         self.col) )
                eof = True
                self.col = 0
                break

            elif freetext:
                break
            
            elif self.blanklineRx.match(line):
                pass

            else:
                m = self.indentRx.match(line)
                indent = m.end()
                self.doIndent(indent)
                self.col = m.end()
                break

        self.lineno = lineno
        if line.endswith("\n"):
            self.currentline = line
        else:
            self.currentline = line + "\n"

        return not eof
    # }}}

    # {{{ def nextToken(self):
    def nextToken(self):
        if self.tokenqueue == []:
            self.readToken()

        return self.tokenqueue[0][0]
    # }}}

    # {{{ def nextOperator(self):
    def nextOperator(self):
        t = self.nextToken()
        if t == failed:
            return failed
        else:
            return self.currentLanguage().getOp(t.text)
    # }}}

    # {{{ def consumeToken(self):
    def consumeToken(self):
        t = self.nextToken()
        _, tcol = self.getPos()
        if t == self.endOfBlock:
            del self.blocks[-1]
        del self.tokenqueue[0]

        if t != failed and not isinstance(t, LayoutToken):
            self.col = tcol + len(t.text)
        return t
    # }}}

    # {{{ def readToken(self):
    def readToken(self):
        line = self.currentline
        linelen = len(line)
        col = self.col

        if col < linelen:
            # {{{ maybe skip whitespace
            m = self.whitespaceRx.match(line, col)
            if m:
                col = m.end()
            # }}}

        if col >= linelen or self.commentRx.match(line, col):
            # {{{ read next line
            if not self.nextLine(False):
                # EOF
                return
            line = self.currentline
            col = self.col
            # }}}
            
        m = self.currentLanguage().matchToken(line, col)
        if m:
            text = line[col:m.end()]
            token = Token(text, None)

        else:
            m = self.stdtokensRx.match(line, col)
            if m:
                for name, text in m.groupdict().items():
                    if text:
                        token = Token(text, name)
                        break
            else:
                token = None

        if token:
            space = (' ', '\t')
            spacenl = (' ', '\t', '\n', '\r')
            
            token.leftSpace = col == 0 or line[col-1] in space
            nextcol = m.end()

            token.rightSpace = nextcol >= len(line) or line[nextcol] in spacenl

            self.tokenqueue.append( (token, self.lineno, col) )
        else:
            self.tokenqueue.append( (failed, self.lineno, col) )
            self.setError("unrecognized symbol (starting %r)" % line[col])
    # }}}

    # {{{ def _getFreetextUntil(self, terminatorRx):
    def _getFreetextUntil(self, terminatorRx):
        m = terminatorRx.search(self.currentline, self.col)
        if m:
            text = self.currentline[self.col:m.start()]
        else:
            text = self.currentline[self.col:]
            while m is None:
                ok = self.nextLine(True)
                if not ok:
                    e = ParseError("EOF inside freetext")
                    self.setError(e.message)
                    raise e

                m = terminatorRx.search(self.currentline)
                if m:
                    text += self.currentline[:m.start()]
                else:
                    text += self.currentline

        self.tokenqueue = []
        return text, m
    # }}}

    # {{{ def freetextUntil(self, terminatorRx):
    def freetextUntil(self, terminatorRx):
        try:
            text, m = self._getFreetextUntil(terminatorRx)
        except ParseError:
            return failed

        for i in range(1, len(m.groups())+1):
            gstart = m.start(i)
            if gstart != -1:
                self.col = gstart
                break
        else:
            self.col = m.end()

        return text
    # }}}

    # {{{ def optextUntil(self, terminatorRx):
    def optextUntil(self, terminatorRx):
        "returns (text, reachedTerminator?)"
        text, m = self._getFreetextUntil(terminatorRx)

        gstart = m.start(1)
        if gstart == -1:
            # Matched an operator, set col to the start of the operator
            self.col = m.start()
            return text, False
        else:
            # Matched the terminator, set col to just after it
            self.col = m.end()
            return text, True
    # }}}

    # {{{ def freetext(self, rx):
    def freetext(self, rx):
        if (len(self.tokenqueue) > 0
            and self.tokenqueue[0][0] == Tokenizer.endOfLine):
            return failed
        
        start = self.col
        line = self.currentline[:-1] # drop trailing newline
        m = rx.match(line, start)
        if m is None:
            return failed
        else:
            self.tokenqueue = []
            self.col = m.end()
            if m.end() == len(line):
                self.nextLine(False)
            return line[start:m.end()]
    # }}}

    # {{{ def startLanguage(self, language):
    def startLanguage(self, language):
        self.languageStack.append(language)
        newqueue = list(itools.takewhile(lambda (t,l,c): isinstance(t, LayoutToken),
                                         self.tokenqueue))
        self.tokenqueue = newqueue
    # }}}

    # {{{ def endLanguage(self):
    def endLanguage(self):
        self.languageStack.pop()
        newqueue = list(itools.takewhile(lambda (t,l,c): isinstance(t, LayoutToken),
                                         self.tokenqueue))
        self.tokenqueue = newqueue
    # }}}

    # {{{ def currentLanguage(self):
    def currentLanguage(self):
        return self.languageStack[-1]
    # }}}

    # {{{ def outerLanguage(self):
    def outerLanguage(self):
        # [-2::-1] gives a list with the last element dropped, and then reversed
        for lang in self.languageStack[-2::-1]:
            if lang != self.currentLanguage():
                return lang
        return self.currentLanguage()
    # }}}

    # {{{ def doIndent(self, indent):
    def doIndent(self, indent):
        blocks = self.blocks

        for ind in blocks[::-1]:
            if indent == ind:
                self.tokenqueue.append( (self.endOfLine, self.lineno, self.col) )
                break
            
            elif indent < ind:
                self.tokenqueue.append( (self.endOfBlock, self.lineno, self.col) )
                
            else:
                break
    # }}}

    # {{{ def startBlock(self):
    def startBlock(self):
        blocks = self.blocks

        t = self.nextToken()
        lineno, col = self.getPos()
 
        if isinstance(t, Token):
            blocks.append(col)

            m = self.indentRx.match(self.currentline)
            if m and m.end() == col:
                return Tokenizer.NEXT_LINE_BLOCK
            else:
                return Tokenizer.SAME_LINE_BLOCK
        else:
            return False
    # }}}

    # {{{ def cancelBlock(self):
    def cancelBlock(self):
        if self.blocks:
            self.blocks.pop()
    # }}}

    # {{{ def finalTerminator(self):
    def finalTerminator(self, op=None, pb=None):
        token = self.nextToken()
        return token is Tokenizer.endOfFile or token is failed
    # }}}

    # {{{ set/clear/backTo Mark
    def setMark(self):
        # ensure the queue is current
        t = self.nextToken()
        
        if isinstance(t, LayoutToken):
            queue = self.tokenqueue
            col = self.col
        else:
            _, col = self.getPos()
            queue = []

        self.marks.append((self.lineno,
                           col,
                           self.linestart,
                           self.blocks[:],
                           queue[:],
                           ))

    def clearMark(self):
        self.marks.pop()

    def backToMark(self):
        m = self.marks[-1]
        (self.lineno, self.col, mlinestart, mblocks, mtokenqueue) = m
        self.blocks = mblocks[:]
        self.tokenqueue = mtokenqueue[:]

        if self.linestart != mlinestart:
            self.linestart = mlinestart
            self.input.seek(self.linestart)
            self.currentline = self.input.readline()

    def breakMark(self):
        assert 0, "breakMark disabled"
        if self.marks != []:
            self.marks[-1] = None
    # }}}

    # {{{ def atEOF(self):
    def atEOF(self):
        return self.tokenqueue[-1][0] == Tokenizer.endOfFile
    # }}}

    # {{{ def getBlockText(self):
    def getBlockText(self):
        filepos = self.input.tell()
        line = self.currentline
        blockindent = indent = self.indentRx.match(line).end()
        lines = []
        blanklines = []
        while 1:
            if line == '':
                break
            elif self.blanklineRx.match(line):
                blanklines.append(line)
            else:
                indent = self.indentRx.match(line).end()
                if indent < blockindent:
                    break
                else:
                    lines.extend(blanklines)
                    blanklines = []
                    lines.append(line)
                
            line = self.input.readline().expandtabs()
            
        self.input.seek(filepos)
        return ''.join(lines)
    # }}}

    # {{{ def skipLines(self, blockText):
    def skipBlock(self, blockText):
        numlines = blockText.count("\n") - 1
        self.tokenqueue = []
        for x in range(numlines):
            self.nextLine(True)
        self.col = len(self.currentline)
    # }}}

    # {{{ def setError(self, )
    def setError(self, message, lineno=False, col=False, language=False):
        if language is False:
            language = self.currentLanguage()

        tline, tcol = self.getPos()

        if lineno is False:
            lineno = tline

        if col is False:
            col = tcol

        furthest = self.furthestError
        if furthest is None:
            self.furthestError = ParseError(message, lineno, col, language)
            
        elif (lineno > furthest.lineno
              or (lineno == furthest.lineno and col > furthest.col)):
            self.furthestError = ParseError(message, lineno, col, language)
            
        elif lineno == furthest.lineno and col == furthest.col:
            self.furthestError = ParseError("syntax error", lineno, col)
    # }}}

    # {{{ def getPos(self):
    def getPos(self):
        if self.tokenqueue == []:
            self.readToken()

        return self.tokenqueue[0][1], self.tokenqueue[0][2]
    # }}}
# }}}
# }}}

# {{{ class OperatorSyntax:
class OperatorSyntax:

    # {{{ def __init__(self, token, binding, ruledef, assoc='left'...):
    def __init__(self, token, binding, ruledef, assoc='left', smartspace=False):
        assert isinstance(binding, int) and binding >= 0
        self.token = token
        self.binding = binding
        self.assoc = assoc or 'left'
        self.smartspace = smartspace

        if type(ruledef) == tuple:
            leftRule, rightRule = ruledef
        else:
            # {{{ extract left and right rules from single rule
            import parser
            islit = lambda x: isinstance(x, parser.LiteralRule)

            if islit(ruledef):
                # non-op (no lhs or rhs)
                leftRule = None
                ruletoken = ruledef.token
                rightRule = None
        
            elif isinstance(ruledef, parser.SequenceRule):
                rules = ruledef.sequence
                if islit(rules[0]):
                    # prefix op
                    leftRule = None
                    ruletoken = rules[0].token
                    rightrules = rules[1:]
                else:
                    # infix or postfix
                    leftRule = rules[0]
                    assert islit(rules[1])
                    ruletoken = rules[1].token
                    rightrules = rules[2:]

                if len(rightrules) == 1:
                    rightRule = rightrules[0]
                elif len(rightrules) == 0:
                    rightRule = None
                else:
                    rightRule = parser.SequenceRule(*rightrules)
            else:
                raise parser.RuleError("invalid syntax rule in defop")

            assert token == ruletoken
            # }}}

        # {{{ raise RuleError if leftRule is invalid
        def exprOrOptExpr(r):
            return (isinstance(r, ExpressionRule)
                    or isinstance(r, Opt) and isinstance(r.rule, ExpressionRule))
        if not (leftRule is None
                or exprOrOptExpr(leftRule)
                or (isinstance(leftRule, NamedRule)
                    and exprOrOptExpr(leftRule.rule))):
            raise RuleError("Invalid left rule %s" % leftRule)
        # }}}

        self.leftRule = leftRule
        self.rightRule = rightRule

        def ruleError(name):
            raise RuleError("Duplicate name '%s' in syntax for operator '%s'" %
                            (name, token))

        lname = getattr(leftRule, 'name', None)
        rname = getattr(rightRule, 'name', None)
        if leftRule and rightRule and lname:
            if lname == rname:
                ruleError(lname)
            elif (not rname) and isinstance(rightRule, parser.SequenceRule):
                for rule in rightRule.sequence:
                    name = isinstance(rule, parser.NamedRule) and rule.name
                    if name == lname: ruleError(name)

        self.rightIsComplex = (not (isinstance(rightRule,
                                               (TermRule, ExpressionRule))
                                    or (isinstance(rightRule, Opt)
                                        and isinstance(rightRule.rule,
                                                       (TermRule,
                                                        ExpressionRule)))))
    # }}}

    def associatesLeft(self):
        return self.assoc == 'left'

    def isListOp(self):
        return self.assoc == 'list'

    # {{{ def packed(self, token):
    def packed(self, token):
        if not self.smartspace:
            return False

        # space on neither side
        if not (token.leftSpace or token.rightSpace):
            return True

        rightExempt = (self.rightRule is None
                       or isinstance(self.rightRule, Opt)
                       or self.rightIsComplex)

        if (not token.leftSpace) and rightExempt:
            return True

        leftExempt = (self.leftRule is None
                      or isinstance(self.leftRule, Opt))

        if (not token.rightSpace) and leftExempt:
            return True

        if leftExempt and rightExempt:
            return True

        return False
    # }}}

    def packedLeft(self, token):
        return self.smartspace and not token.leftSpace

    def packedRight(self, token):
        return self.smartspace and not token.rightSpace

    def isPrefixOp(self, token):
        return (self.leftRule is None
                or (isinstance(self.leftRule, Opt)
                    and self.packed(token)
                    and not self.packedLeft(token)))

    def isEnclosed(self):
        rr = self.rightRule
        return rr is None or rr.isEnclosing()
# }}}

# {{{ Parse Rules
# {{{ nothing = 
class Nothing:
    def __repr__(self): return "<parser.nothing>"
nothing = Nothing()
# }}}

# {{{ nothing = 
class Outer:
    def __repr__(self): return "<parser.outer>"
outer = Outer()
# }}}

# {{{ def tokenToValue(token, tokens):
def tokenToValue(tokens):
    
    token = tokens.nextToken()
    t = token.text

    if token.value in ("string", "number"):
        res = eval(t,{})

    else:
        assert not tokens.currentLanguage().hasop(t)

        if token.value == 'symbol':
            res = Symbol(t)
        else:
            tokens.setError('unexpected "%s"' % t)
            return failed

    tokenmacro = tokens.currentLanguage().getTokenMacro(token.value)
    if tokenmacro:
        return tokenmacro(res)
    else:
        return res
# }}}

# {{{ def operatorArgs(body):
def operatorArgs(body):

    if body == nothing:
        res = flist()

    elif isinstance(body, list):
        res = flist()
        for x in body:
            fl = operatorArgs(x)

            # {{{ error if duplicate field name
            for n in fl.fields.keys():
                if res.hasField(n):
                    raise RuleError("Duplicate field: %s" % n)
            # }}}

            res.extend(fl)

    elif type(body) == tuple:
        name,val = body
        if isinstance(val, tuple):
            raise RuleError("conflicting names")
        elif isinstance(val, list):
            x = operatorArgs(val)
        else:
            x = val

        if name:
            res = flist({name:x})
        else:
            res = flist([x])
        
    else:
        res = flist([body])

    return res
# }}}

# {{{ def parseOperator(...):
def parseOperator(lhs, tokens, optoken, operator, isTerminator, execenv, opPos):
    lineno = tokens.lineno

    opsyn = operator.__syntax__
    rrule = opsyn.rightRule
    lrule = opsyn.leftRule

    if optoken and opsyn.isPrefixOp(optoken):
        opPrefixBinding = opsyn.binding
    else:
        opPrefixBinding = None

    # {{{ def terminator(nextOp=None):
    def terminator(nextOp=None, prefixBinding=None):
        # {{{ return True if parent terminator says so
        if prefixBinding is None:
            prefixBind = opPrefixBinding
        elif opPrefixBinding is None:
            prefixBind = prefixBinding
        else:
            prefixBind = min(opPrefixBinding, prefixBinding)
        if isTerminator(nextOp, prefixBind):
            return True
        # }}}

        if nextOp: # continuation operator - no token
            nextToken = None
        else:
            nextToken = tokens.nextToken() 
            nextOp = tokens.nextOperator()

        if nextOp is failed:
            return True
        if nextOp:
            # Encountered a new operator while parsing rhs of \operator
            # return True to stop parsing rhs of \operator
            #    i.e. \operator and its rhs become the lhs sub-expr
            #    of \nextOp
            # return False to keep parsing rhs of \operator
            #    i.e. \nextOp becomes a sub-expression in the rhs
            #    of \operator

            nextSyn = nextOp.__syntax__

            # prefix operator always starts sub-expression
            if nextSyn.isPrefixOp(nextToken):
                return False

            # if nextOp is inside any prefix operator, it cannot
            # terminate this operator unless it binds looser than
            # that prefix op
            if prefixBinding is not None and nextSyn.binding > prefixBinding:
                return False

            # end operator here if next operator binds more loosely
            # continue if it binds more tightly

            # {{{ determin if current and next are packed
            currentPacked = opsyn.packed(optoken)
            if opsyn.smartspace and not currentPacked:
                # current operator is not allowed to pack
                # so neither is next
                nextPacked = False
            else:
                nextPacked = nextSyn.packed(nextToken)
            # }}}

            if currentPacked and not nextPacked:
                return True
            elif not currentPacked and nextPacked:
                return False
            elif (nextSyn.binding < opsyn.binding
                  or (nextSyn.binding == opsyn.binding
                      and opsyn.associatesLeft())):
                return True
            else:
                return False

        else:
            return False
    # }}}

    if lrule:
        # {{{ do lhs
        if lhs == None:
            if isinstance(lrule, Opt):
                lhs = lrule.alt
            elif (isinstance(lrule, NamedRule) and
                  isinstance(lrule.rule, Opt)):
                lhs = lrule.rule.alt
            else:
                tokens.setError('Missing LHS for %s' % opsyn.token, *opPos)
                return failed
        # }}}

    if rrule:
        # {{{ parse rhs (if error, return it)
        if (isinstance(rrule, Opt)
            and opsyn.packed(optoken)
            and not opsyn.packedRight(optoken)):
            rhs = None
        else:
            rhs = rrule.parse(tokens, terminator, execenv)
            if rhs == failed:
                return failed
        # }}}

    fields = {}
    
    if lrule and lhs != nothing:
        if isinstance(lrule, NamedRule):
            operands = flist(**{lrule.name:lhs})
        else:
            operands = flist([lhs])
    else:
        operands = flist()

    if rrule:
        operands += operatorArgs(rhs)

    res = instantiateOp(operator, operands, lineno)

    return res
# }}}

# {{{ class Rule:
class Rule(object):

    def mightStartWith(self):
        "a set of strings - a match might start with one of them"
        return []

    def allLiterals(self):
        return []

    def mustStartWith(self):
        "a set of tokens - a match will start with one of them"
        return []

    def matchesZeroTokens(self):
        return False

    def requiresLiterals(self):
        return False

    def isEnclosing(self):
        return False

    def hasValue(self):
        return True

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.pprint())
# }}}

# {{{ class LocalLangRule(Rule):
class LocalLangRule(Rule):

    def __init__(self, language=None):
        if language == "^":
            self.language = outer
        else:
            assert language is None or isinstance(language, Language)
            self.language = language and language.__impl__

    def parse(self, tokens, isTerminator, execenv):
        if self.language and self.language != tokens.currentLanguage():
            if self.language is outer:
                tokens.startLanguage(tokens.outerLanguage())
            else:
                tokens.startLanguage(self.language)
            res = self._parse(tokens, isTerminator, execenv)
            tokens.endLanguage()
            return res
        else:
            return self._parse(tokens, isTerminator, execenv)
# }}}
        
# {{{ class NamedRule(Rule):
class NamedRule(Rule):

    def __init__(self, name, rule):
        if isinstance(rule, (NamedRule, ParsedNameRule)):
            raise RuleError("cannot name named rule")
        assert type(name) == str or name is None
        self.name = name
        self.rule = rule
        
    def parse(self, *args, **options):
        res = self.rule.parse(*args, **options)

        if res in (failed, nothing):
            return res
        else:
            if isinstance(self.rule, Rep1):
                # make a list of lists
                vals = []
                for x in res:
                    if isinstance(x, list):
                        vals.append( (None, x) )
                    else:
                        vals.append( x)
                return (self.name, vals)
            else:
                return (self.name, res)

    def matchesZeroTokens(self):
        return self.rule.matchesZeroTokens()

    def mightStartWith(self):
        return self.rule.mightStartWith()

    def mustStartWith(self):
        return self.rule.mustStartWith()  

    def allLiterals(self):
        return self.rule.allLiterals()

    def requiresLiterals(self):
        return self.rule.requiresLiterals()

    def isEnclosing(self):
        return self.rule.isEnclosing()

    def pprint(self): return '$%s:%s' % (self.name, self.rule.pprint())
# }}}

# {{{ class TermRule(LocalLangRule):
class TermRule(LocalLangRule):

    # {{{ def _parse(self, tokens, isTerminator, execenv):
    def _parse(self, tokens, isTerminator, execenv):
        continueOp = self.getContinuationOp(tokens.currentLanguage())
        if continueOp:
            assert isinstance(continueOp.__syntax__.leftRule, ExpressionRule)

        if isTerminator():
            tokens.setError("expected %s" % self.errorName())
            term = failed
        else:
            token = tokens.nextToken()
            tokenPos = tokens.getPos()
            operator = tokens.nextOperator()

            if operator:
                tokens.consumeToken()
                term = parseOperator(None, tokens, token, operator,
                                     isTerminator, execenv, tokenPos)
            else:
                term = tokenToValue(tokens)
                tokens.consumeToken()

        # {{{ try to continue the term with infix ops and maybe continuation op
        useContinueOp = bool(continueOp)
        while term != failed:
            token = tokens.nextToken()
            if token == failed:
                break
            
            tokenOp = tokens.nextOperator()
            tokenPos = tokens.getPos()

            # need an infix op to continue
            opsyn = tokenOp and tokenOp.__syntax__
            if tokenOp and not opsyn.isPrefixOp(token):
                if isTerminator():
                    break
                else:
                    tokens.consumeToken()
                    operator = tokenOp
                    opToken = token
                    useContinueOp = True
            elif useContinueOp and continueOp and not isTerminator(continueOp):
                operator = continueOp
                useContinueOp = False
                opToken = None
            else:
                # no joining operator to continue with
                break

            term = parseOperator(term, tokens, opToken, operator,
                                 isTerminator, execenv, tokenPos)
        # }}}
                
        return term # might be \failed
    # }}}

    def getContinuationOp(self, language):
        return None

    def pprint(self): return 'term'

    def errorName(self): return 'term'
# }}}

# {{{ class ExpressionRule(TermRule): 
class ExpressionRule(TermRule):

    def getContinuationOp(self, language):
        return language.getContinuationOp()
    
    def pprint(self): return 'expr'

    def errorName(self): return 'expression'
# }}}

# {{{ class BlockRule(LocalLangRule): 
class BlockRule(LocalLangRule):

    lineParser = ExpressionRule()

    # {{{ def _parse(self, tokens, language, isTerminator):
    def _parse(self, tokens, isTerminator, execenv):
        currentLang = tokens.currentLanguage()

        # {{{ lines = do the parse
        blockType = tokens.startBlock()

        lines = None
        error = False
        blocktext = None
        blocklineno = tokens.lineno
        if currentLang.name.endswith("~"):
            blocklang = currentLang.parent
        else:
            blocklang = currentLang
        opcount = len(blocklang.operators)
            
        # {{{ try to get lines from cache
        if blockType == Tokenizer.NEXT_LINE_BLOCK:
            blocktext = tokens.getBlockText()
            
            lines = blocklang.blockCache(blocktext, blocklineno)
            if lines is not None:
                tokens.cancelBlock()
                tokens.skipBlock(blocktext)
        # }}}

        if blockType is not False and lines == None:
            lines = []
            
            # {{{ terminator = select terminator func for block type
            def sameLineTerminator(op=None, prefixBinding=None):
                t = tokens.nextToken()
                return (isTerminator(op, prefixBinding)
                        or isinstance(t, LayoutToken)
                        or t == failed)

            def bodyTerminator(op=None, prefixBinding=None):
                t = tokens.nextToken()
                return isinstance(t, LayoutToken) or t == failed
            
            terminator = blockType == Tokenizer.SAME_LINE_BLOCK and \
                         sameLineTerminator or bodyTerminator
            # }}}
            
            # {{{ main parsing loop
            languageChanged = False
            running = True
            while running:
                lineno = tokens.lineno

                line = BlockRule.lineParser.parse(tokens, terminator, execenv)

                if isinstance(line, Symbol):
                    setmeta(line, lineno=lineno)

                tok = tokens.nextToken()
                
                if tok == failed or line == failed:
                    error = True
                    tokens.cancelBlock()
                    break

                if tok in (Tokenizer.endOfLine, Tokenizer.endOfBlock):
                    tokens.consumeToken()
                elif not terminator():
                    tokens.setError("unexpected %r" % tok.text)
                    error = True
                    break

                appendline, newlang = self.processLine(line, execenv, tokens)
                if newlang is not None and newlang.__impl__ != currentLang:
                    if languageChanged:
                        tokens.endLanguage()
                    tokens.startLanguage(newlang.__impl__)
                    currentLang = newlang.__impl__
                    languageChanged = True
                if appendline is not None:
                    lines.append(appendline)

                if tok == Tokenizer.endOfBlock:
                    break
                elif terminator():
                    tokens.cancelBlock()
                    break

                # always use bodyTerminator after first line
                terminator = bodyTerminator

            if languageChanged:
                tokens.endLanguage()
            # }}}
                
            if ((not error)
                and blocktext
                and (not languageChanged) # this is inadequate -
                                          # doesn't catch setlang in a sub-block
                and len(blocklang.operators) == opcount # no defops in block
                and blocktext.count("\n") > 1 # don't cache one-liners
                ):
                blocklang.setBlockCache(blocktext, lines, blocklineno)
        # }}}

        if error:
            return failed
        else:
            return lines or []
    # }}}

    def pprint(self): return 'block'

    # {{{ def processLine(self, line, execenv, tokens):
    def processLine(self, line, execenv, tokens):
        lineno=getmeta(line, "lineno")
        if isinstance(line, rootops.deflang):
            raise ParseError("deflang must be at the top-level", lineno)
        
        elif isinstance(line, rootops.defop):
            raise ParseError("defop must be at the top-level or in a deflang",
                             lineno)
        
        elif isinstance(line, rootops.getops):
            raise ParseError("getops must be at the top-level or in a deflang",
                             lineno)
        
        elif execenv is not None and isinstance(line, rootops.setlang):
            expline = macros.expand(line)
            code = compiler.compile([expline], tokens.filename, mode='eval',
                                    module=execenv['__name__'])
            newlang = eval(code, execenv)
            if not isinstance(newlang, Language):
                raise ParseError("setlang to non-language: %r" % newlang, lineno)
            
            return None, newlang
        
        else:
            return line, None
    # }}}
# }}}

# {{{ class TopLevelBlockRule(BlockRule):
class TopLevelBlockRule(BlockRule):

    """Specializes BlockRule in order to handle defop and deflang at the
       top-level a source file. There are no syntactic changes.
    """

    def parse(self, tokens, isTerminator, execenv):
        assert execenv is not None
        self.codeobjects = []
        result = BlockRule.parse(self, tokens, isTerminator, execenv)
        return result, self.codeobjects

    # {{{ def processLine(self, line, execenv, tokens):
    def processLine(self, line, execenv, tokens):
        "Returns (line-to-be-appended-to-result-or-None, new-language-or-None)"

        line2 = macros.expand(line)

        # {{{ maybe add custom attributes to the line
        # don't check \line2 for setlang - not honored in macro expansions
        if isinstance(line, rootops.setlang):
            line2['temp'] = True
        elif isinstance(line2, (rootops.defop, rootops.getops)):
            currentlang = tokens.currentLanguage()
            langmod = currentlang.userlang.__module__
            if execenv is None or langmod == execenv['__name__']:
                lang = currentlang.name
            else:
                assert 0, ("PROBE: Why would we be adding ops to a language"
                           "from another module?")
                lang = "%s.%s" % (langmod, currentlang.name)
            line2['lang'] = lang
        # }}}

        code = compiler.compile([line2], tokens.filename, mode='eval',
                                module=execenv['__name__'])
        if isinstance(line2, rootops.deflang):
            # Don't eval (already done by DeflangBlockRule)
            # but do include in final codeobject
            self.codeobjects.append(code)
            return line2, None

        else:
            lineResult = eval(code, execenv)

            # setlang is not honored in a macro expansion
            # so check \line rather than \line2
            if isinstance(line, rootops.setlang):
                # Keep hold of the temporary lang in the environment - 
                # It's needed by any following top-level defops
                newlang = lineResult
                if not isinstance(newlang, Language):
                    raise ParseError("setlang to non-language: %r" % newlang,
                                     lineno=getmeta(line, 'lineno'))

                execenv[newlang.__impl__.name] = newlang
                self.codeobjects.append(code)
                return line2, newlang
            elif isinstance(line2, rootops.getops):
                # A top-level getops is only
                # a parse-time pragma -
                # omit from final codeobject
                return None, None
            else:
                self.codeobjects.append(code)
                return line2, None
    # }}}
# }}}

# {{{ class LiteralRule(Rule):
class LiteralRule(Rule):

    def __init__(self, token):
        self.token = token

    def parse(self, tokens, isTerminator, execenv):
        token = tokens.nextToken()
        # isinstance check so we can never match a layout token
        if isinstance(token, Token) and self.token == token.text:
            tokens.consumeToken()
            return token.text
        elif token == failed:
            return failed
        else:
            tokens.setError('expected "%s"' % self.token)
            return failed

    def mightStartWith(self):
        return [Token(self.token)]

    def mustStartWith(self):
        return [Token(self.token)]

    def allLiterals(self):
        return [self.token]

    def requiresLiterals(self):
        return True

    def hasValue(self):
        return False

    def isEnclosing(self):
        return True

    def pprint(self): return '"%s"' % self.token
# }}}

# {{{ class TokenRule(LocalLangRule):
class TokenRule(LocalLangRule):

    def _parse(self, tokens, isTerminator, execenv):
        token = tokens.nextToken()

        # isinstance check so we can never match a layout token
        if isinstance(token, Token):
            operator = tokens.nextOperator()
            if operator:
                res = operator
            else:
                res = tokenToValue(tokens)
            tokens.consumeToken()
            
        elif token == failed:
            # error already set by tokenizer
            res = failed
        else:
            tokens.setError('expected a token')
            res = failed

        return res

    def pprint(self): return 'token'
# }}}

# {{{ class SymbolRule(Rule):
class SymbolRule(Rule):

    termRule = TermRule()

    def parse(self, tokens, isTerminator, execenv):
        if isTerminator():
            tokens.setError('expected a symbol')
            return failed
        
        token = tokens.nextToken()

        escapeSyn = getattr(rootops.escape, '__syntax__', None)
        if escapeSyn and token.text == escapeSyn.token:
            pos = tokens.getPos()
            tokens.consumeToken()
            return parseOperator(None, tokens, token, rootops.escape,
                                 isTerminator, execenv, pos)
        else:
            if (isinstance(token, Token)
                and token.value == 'symbol'):
                tokens.consumeToken()
                return Symbol(token.text)
            else:
                tokens.setError('expected a symbol')
                return failed

    def pprint(self): return 'symbol'
# }}}

# {{{ class SequenceRule(Rule):
class SequenceRule(Rule):

    # {{{ def __init__(self, *sequence):
    def __init__(self, *sequence):
        self.sequence = sequence

        withvalues = [r for r in sequence if r.hasValue()]
        self.singleton = len(withvalues) == 1

        self.termtokens = [self.nextTokens(i)
                           for i in range(len(sequence))]
    # }}}
            
    # {{{ def parse(self, tokens, isTerminator, execenv):
    def parse(self, tokens, isTerminator, execenv):
        lst = []
        seqlen = len(self.sequence)
        termtokens = self.termtokens
        rules = self.sequence
        for i,rule in enumerate(rules):
            # {{{ def terminator(op=None):
            def terminator(op=None, prefixBinding=None):
                token = tokens.nextToken()
                
                if token == failed:
                    return True

                notend = i+1 < seqlen
                if notend and self.requiresLiteralsAt(i+1):
                    # expecting more tokens in this seq - only pass
                    # LayoutTokens to parent terminator...
                    if isinstance(token, LayoutToken) and isTerminator(op):
                        return True
                    
                else:
                    if isTerminator(op, prefixBinding):
                        return True

                if notend:
                    return token in termtokens[i+1]
                else:
                    return False
            # }}}

            val = rule.parse(tokens, terminator, execenv)

            if val is failed:
                return failed

            if val == nothing or isinstance(rule, LiteralRule):
                pass # omit from sequence
            else:
                lst.append(val)

        if self.singleton:
            if lst != []:
                res = lst[0]
            else:
                res = nothing
        else:
            res = lst
        return res
    # }}}

    # {{{ def nextTokens(self, index):
    def nextTokens(self, index):
        "A set of tokens the sequence is expecting at \index"
        seq = self.sequence
        seqlen = len(seq)
        toks = seq[index].mightStartWith()
        i = index
        while seq[i].matchesZeroTokens() and i+1 < seqlen:
            i += 1
            toks.extend(seq[i].mightStartWith())
        return toks
    # }}}

    # {{{ def mightStartWith(self):
    def mightStartWith(self):
        return self.nextTokens(0)
    # }}}

    # {{{ def mustStartWith(self):
    def mustStartWith(self):
        for rule in self.sequence:
            if isinstance(rule, TrivialRule):
                pass
            else:
                return rule.mustStartWith()
    # }}}

    # {{{ def allLiterals(self):
    def allLiterals(self):
        return concat([r.allLiterals() for r in self.sequence])
    # }}}

    # {{{ def requiresLiteralsAt(self, index):
    def requiresLiteralsAt(self, index):
        for rule in self.sequence[index:]:
            if rule.requiresLiterals():
                return True
        return False
    # }}}

    # {{{ def requiresLiterals(self):
    def requiresLiterals(self):
        return self.requiresLiteralsAt(0)
    # }}}

    # {{{ def hasValue(self):
    def hasValue(self):
        for x in self.sequence:
            if x.hasValue(): return True
        return False
    # }}}

    # {{{ def pprint(self):
    def pprint(self):
        return '(%s)' % ' '.join([r.pprint() for r in self.sequence])
    # }}}

    # {{{ def isEnclosing(self):
    def isEnclosing(self):
        for r in self.sequence[::-1]:
            if r.isEnclosing():
                return True
            elif isinstance(r, TrivialRule):
                # Ignore this one - check previous
                pass
            else:
                return False
    # }}}
# }}}

# {{{ class ChoiceRule(Rule):
class ChoiceRule(Rule):

    # {{{ def __init__(self, *choices):
    def __init__(self, *choices):
        self.choices = choices
        self._mustStarts = [c.mustStartWith() for c in choices]
    # }}}
    
    # {{{ def parse(self, tokens, isTerminator, execenv):
    def parse(self, tokens, isTerminator, execenv):
        tok = tokens.nextToken()
        if tok == failed:
            return failed
        
        choices = [rule for rule, mustStarts in zip(self.choices, self._mustStarts)
                   if mustStarts == [] or tok in mustStarts]
        if len(choices) == 0:
            tokens.setError("unexpected %r" % tok.text)
            val = failed
        elif len(choices) == 1:
            val= choices[0].parse(tokens, isTerminator, execenv)
        else:
            tokens.setMark()
            for rule in choices:
                val = rule.parse(tokens, isTerminator, execenv)
                if val == failed:
                    tokens.backToMark()
                else:
                    break
            else:
                tokens.setError("syntax error")
                val = failed
            tokens.clearMark()

        return val
    # }}}

    # {{{ def mustStartWith(self):
    def mustStartWith(self):
        lits = []
        for rule in self.choices:
            starts = rule.mustStartWith()
            if starts == []:
                return []
            else:
                lits.extend(starts)
        return lits
    # }}}
    
    # {{{ def mightStartWith(self):
    def mightStartWith(self):
        return concat([r.mightStartWith() for r in self.choices])
    # }}}

    # {{{ def allLiterals(self):
    def allLiterals(self):
        return concat([r.allLiterals() for r in self.choices])
    # }}}

    # {{{ def requiresLiterals(self):
    def requiresLiterals(self):
        for x in self.choices:
            if not x.requiresLiterals():
                return False
        return True
    # }}}

    # {{{ def isEnclosing(self):
    def isEnclosing(self):
        for x in self.choices:
            if not x.isEnclosing():
                return False
        return True
    # }}}

    # {{{ def hasValue(self):
    def hasValue(self):
        for x in self.choices:
            if x.hasValue(): return True
        return False
    # }}}

    # {{{ def matchesZeroTokens(self):
    def matchesZeroTokens(self):
        for r in self.choices:
            if r.matchesZeroTokens(): return True
        return False
    # }}}

    # {{{ def pprint(self):
    def pprint(self):
        return '(%s)' % ' | '.join([r.pprint() for r in self.choices])
    # }}}
# }}}

# {{{ class TrivialRule(Rule):
class TrivialRule(Rule):

    def __init__(self, *args):
        if len(args) > 0:
            self.result = args[0]
        else:
            self.result = nothing

    def parse(self, *args, **options):
        return self.result

    def pprint(self):
        if self.result is nothing:
            return '<>'
        else:
            return '<%s>' % self.result

    def matchesZeroTokens(self):
        return True
# }}}

# {{{ class DebugRule(TrivialRule):
class DebugRule(TrivialRule):

    def __init__(self, message):
        self.message = message

    def parse(self, *args, **options):
        if self.message == None:
            debug()
        else:
            print 'parse debug:', self.message
        return nothing

    def hasValue(self):
        return False

    def pprint(self):
        return 'debug(%s)' % self.message

    def matchesZeroTokens(self):
        return True
# }}}

# {{{ class FreetextRule(Rule):
class FreetextRule(Rule):

    def __init__(self, regex, upto):
        self.regex = regex
        self.rx = re.compile(regex)
        self.upto = upto

    def parse(self, tokens, isTerminator, execenv):
        if self.upto:
            return tokens.freetextUntil(self.rx)
        else:
            return tokens.freetext(self.rx)

    def pprint(self):
        return 'freetext %s/%s/' % (self.upto and "upto " or "", self.regex)

    def isEnclosing(self):
        return True
# }}}

# {{{ class OptextRule(LocalLangRule):
class OptextRule(LocalLangRule):

    def __init__(self, terminator, language=None):
        LocalLangRule.__init__(self, language)
        self.terminator = terminator

    def _parse(self, tokens, isTerminator, execenv):
        exp = ExpressionRule()
        
        rx = tokens.currentLanguage().optextRegex(self.terminator)

        res = []
        running = True
        while running:
            try:
                text, gotTerminator = tokens.optextUntil(rx)
            except ParseError:
                return failed
            
            if len(text) > 0:
                res.append(text)
            
            if gotTerminator:
                running = False
            else:
                def terminator(op=None, prefixBinding=None):
                    return isinstance(tokens.nextToken(), LayoutToken)
                
                x = exp.parse(tokens, terminator, execenv)
                if x == failed:
                    res = failed
                    running = False
                else:
                    res.append(x)
        
        return res

    def pprint(self):
        return 'freetext /%s/' % self.terminator

    def isEnclosing(self):
        return True
# }}}

# {{{ class EolRule(Rule):
class EolRule(Rule):

    def parse(self, tokens, *args, **options):
        token = tokens.nextToken()

        if token == tokens.endOfLine:
            tokens.consumeToken()
            return nothing
        else:
            if token == failed:
                return failed
            else:
                tokens.setError('expected end-of-line')
                return failed

    def hasValue(self):
        return False

    def mustStartWith(self):
        return [Tokenizer.endOfLine]

    def mightStartWith(self):
        return [Tokenizer.endOfLine]

    def pprint(self): return 'eol'
# }}}

# {{{ class Opt(Rule):
class Opt(Rule):

    def __init__(self, rule, alt=object):
        self.rule = rule

        # I know it's weird that we don't simply use \nothing as the default
        # but don't change it! It's to do with experimental reloading stuff
        self.alt = alt is object and nothing or alt

        self._mustStart = self.rule.mustStartWith()

        
    def parse(self, tokens, isTerminator, execenv):
        if self._mustStart == [] or tokens.nextToken() in self._mustStart:
            tokens.setMark()

            val = self.rule.parse(tokens, isTerminator, execenv)

            if val == failed:
                tokens.backToMark()
                res = self.alt
            else:
                res = val

            tokens.clearMark()
            return res
        else:
            return self.alt

    def matchesZeroTokens(self):
        return True

    def mightStartWith(self):
        return self.rule.mightStartWith()

    def allLiterals(self):
        return self.rule.allLiterals()

    def hasValue(self):
        return self.rule.hasValue() or self.alt is not nothing

    def pprint(self): return '[%s]' % self.rule.pprint()
# }}}

# {{{ class Rep1(Rule):
class Rep1(Rule):

    # {{{ def __init__(self, rule):
    def __init__(self, rule):
        self.rule = rule
        self.mightStart = self.mightStartWith()
    # }}}

    # {{{ def parse(self, tokens, isTerminator, execenv):
    def parse(self, tokens, isTerminator, execenv):
        def terminator(op=None, prefixBinding=None):
            return isTerminator(op, prefixBinding) or \
                   tokens.nextToken() in self.rule.mightStartWith()

        vals = []
        while 1:
            tokens.setMark()
            
            val = self.rule.parse(tokens, terminator, execenv)
            if val == failed:
                tokens.backToMark()
                tokens.clearMark()
                break

            else:
                vals.append(val)
                tokens.clearMark()
            
        if vals == [] and self.__class__ == Rep1:
            return failed
        else:
            return vals
    # }}}

    # {{{ def mustStartWith(self):
    def mustStartWith(self):
        return self.rule.mustStartWith()
    # }}}

    # {{{ def mightStartWith(self):
    def mightStartWith(self):
        return self.rule.mightStartWith()
    # }}}

    # {{{ def allLiterals(self):
    def allLiterals(self):
        return self.rule.allLiterals()
    # }}}

    # {{{ def requiresLiterals(self):
    def requiresLiterals(self):
        return self.rule.requiresLiterals()
    # }}}

    # {{{ def pprint(self):
    def pprint(self):
        return '%s+' % self.rule.pprint()
    # }}}
# }}}

# {{{ class Rep(Rep1):
class Rep(Rep1):

    def mustStartWith(self):
        return []

    def matchesZeroTokens(self):
        return True

    def requiresLiterals(self):
        return False

    def pprint(self): return '%s*' % self.rule.pprint()
# }}}

# {{{ class ParsedNameRule(Rule):
class ParsedNameRule(Rule):

    def __init__(self, rule):
        if isinstance(rule, (NamedRule, ParsedNameRule)):
            raise RuleError("cannot name named rule")
        
        self.rule = rule
        
    def parse(self, *args, **options):
        sym = SymbolRule().parse(*args, **options)
        if sym == failed:
            return failed
        
        res = self.rule.parse(*args, **options)
        if res == failed:
            return failed
        elif res is nothing:
            return nothing
        else:
            return (str(sym), res)

    def allLiterals(self):
        return self.rule.allLiterals()

    def requiresLiterals(self):
        return self.rule.requiresLiterals()

    def isEnclosing(self):
        return self.rule.isEnclosing()

    def pprint(self): return 'symbol:%s' % self.rule.pprint()
# }}}

# {{{ class SwitchlangRule(Rule):
class SwitchlangRule(Rule):
    """A custom parse rule for parsing the expression inside
    a switchlang '(: ... )' operator"""

    def parse(self, tokens, isTerminator, execenv):
        lineno, col = tokens.getPos()
        langx = TermRule().parse(tokens, isTerminator, execenv)

        if execenv is None:
            raise ParseError("cannot switch languages: parse has no execenv",
                             lineno, col)
            
        exp = macros.expand(langx)
        code = compiler.compile([exp], tokens.filename, mode='eval',
                                module=execenv['__name__'])
        lang = eval(code, execenv)

        if not isinstance(lang, Language):
            raise ParseError("not a language: %s" % lang, lineno, col)

        return ExpressionRule(lang).parse(tokens, isTerminator, execenv)

    def pprint(self): return '<<switchlang-rule>>'
# }}}

# {{{ class DeflangRule(Rule):
class DeflangRule(Rule):

    def __init__(self):
        self.rule = SequenceRule(
            SymbolRule(),
            Opt(SequenceRule(LiteralRule("("),
                             ExpressionRule(),
                             LiteralRule(")")), None),
            LiteralRule(":"))
            # NamedRule("body", BlockRule) - it's like this but the block is special
        self.blockRule = DeflangBlockRule()

    def parse(self, tokens, isTerminator, execenv):
        lineno, col = tokens.getPos()
        res = self.rule.parse(tokens, isTerminator, execenv)
        if res == failed:
            return failed
        else:
            if execenv is None:
                raise ParseError("cannot use deflang: parse has no execenv",
                                  lineno, col)
            else:
                langname = str(res[0])
                # {{{ parent = 
                parentx = res[1]
                if parentx == None:
                    parent = rootops.defaultBaseLang
                else:
                    parentx2 = macros.expand(parentx)
                    code = compiler.compile([parentx2], tokens.filename, mode='eval')
                    parent = eval(code, execenv)
                # }}}

                try:
                    lang = defLanguage(langname, parent, execenv)
                except LanguageBaseException, e:
                    raise ParseError(str(e), lineno, col)
                execenv[langname] = lang

                body = self.blockRule.parse(lang, tokens, isTerminator, execenv)

                if body == failed:
                    return failed
                else:
                    return res + [('body',body)]

    def allLiterals(self):
        return ['(', ')', ':']

    def pprint(self): return '<<deflang-rule>>'
# }}}

# {{{ class DeflangBlockRule(BlockRule):
class DeflangBlockRule(BlockRule):

    def parse(self, lang, tokens, isTerminator, execenv):
        if execenv is None:
            raise ParseError("cannot use deflang: parse has no execenv",
                             *tokens.getPos())
        
        self.locals = {}
        self.newlang = lang
        ilang = lang.__impl__

        result = BlockRule.parse(self, tokens, isTerminator, execenv)

        ilang.addDeflangLocals(self.locals)

        return result

    def processLine(self, line, execenv, tokens):
        lineno = getmeta(line, "lineno")
        expline = macros.expand(line)
        
        if isinstance(expline, rootops.deflang):
            raise ParseError("deflang must be at the top-level", lineno)

        elif isinstance(line, rootops.setlang):
            code = compiler.compile([expline], tokens.filename, mode='eval',
                                    module=execenv['__name__'])
            newlang = eval(code, execenv)
            if not isinstance(newlang, Language):
                raise ParseError("setlang to non-language: %r", lineno)
            
            return None, newlang

        # {{{ maybe add custom atributes to expline
        if isinstance(expline, (rootops.defop, rootops.getops)):
            langmod = self.newlang.__module__
            newlangname = self.newlang.__impl__.name
            if langmod == execenv['__name__']:
                lang = newlangname
            else:
                assert 0, ("PROBE: How could the language in a deflang"
                           "come from another module?!")
                lang = "%s.%s" % (langmod, newlangname)
            expline['lang'] = lang
        # }}}

        code = compiler.compile([expline], tokens.filename, mode='eval',
                                module=execenv['__name__'])
        eval(code, execenv, self.locals)

        lang = self.newlang
        for var in LanguageImpl.configVars:
            if (var in self.locals
                and getattr(lang, var, None) != self.locals[var]):
                setattr(lang, var, self.locals[var])

        return expline, None
# }}}

# {{{ class GetopsRule(Rule):
class GetopsRule(Rule):
    """A custom rule for "getops x, a b c". Evaluates x during the parse,
       and then parses tokens in that language. Only terminates on a
       LayoutToken (i.e. getops should only be used at top-level of a line)."""

    def __init__(self):
        self.rule = SequenceRule(ExpressionRule(),
                                 Opt(SequenceRule(LiteralRule(",")), None))

    def parse(self, tokens, isTerminator, execenv):
        lineno, col = tokens.getPos()
        res = self.rule.parse(tokens, isTerminator, execenv)
        if res == failed:
            return failed
        else:
            if res[1] is None:
                return res[0]
            else:
                langx = macros.expand(res[0])
                code = compiler.compile([langx], tokens.filename, mode='eval')
                lang = eval(code, execenv)

                if not isinstance(lang, Language):
                    raise ParseError("expected language, got %r in getops" % lang,
                                     lineno, col)

                operators = []
                tokenRule = TokenRule(lang)
                while not isinstance(tokens.nextToken(), LayoutToken):
                    lineno, col = tokens.getPos()
                    op = tokenRule.parse(tokens, isTerminator, execenv)
                    if isinstance(op, OperatorType):
                        operators.append(op)
                    else:
                        raise ParseError("getops: no operator '%s' in language %s"
                                         % (op, lang.__impl__.name),
                                         lineno, col)

                return [res[0]] + operators

    def allLiterals(self):
        return [',']

    def pprint(self): return '<<getops-rule>>'
# }}}
# }}}

import macros
import rootops
import pycompile as compiler
