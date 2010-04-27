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
import operator
import types
import new

from parser import *
from language import \
     Language, Operator, BaseOperator, BaseMacro, pprint, opfields
import language
from data import Symbol, flist
# }}}

# {{{ Base Operators
# {{{ class CallFunction(BaseOperator):
class CallFunction(BaseOperator):

    def __init__(self, *args, **kws):
        BaseOperator.__init__(self, *args, **kws)
        assert len(list(self.__operands__)) > 0,"invalid function call (no function)"

    def __pprint__(self, indent, parens=False):
        args = self.__operands__
        pp = getattr(args[0], '__pprint__', None)
        if pp:
            head = pp(0, True)
        else:
            head = repr(args[0])
        return pprint(head, args.elems[1:], args.items(),
                      indent, lparen="{", rparen="}")
# }}}

# These all have __syntax__ attributes added later

class switchlang(BaseMacro):

    macro = staticmethod(lambda exp: exp)

class outerlang(BaseMacro):

    macro = staticmethod(lambda exp: exp)

class GetRuleClass(BaseOperator):
    pass

class defop(BaseOperator):
    pass

class deflang(BaseOperator):
    pass

class setlang(BaseOperator):
    pass

class getops(BaseOperator):
    pass

class quote(BaseOperator):
    pass

class opquote(BaseOperator):
    pass

class escape(BaseOperator):
    pass

class localmodule(BaseMacro):
    pass

class PyOp(BaseOperator):
    pass

class PyExp(BaseOperator):
    pass
# }}}

# {{{ rule lang
# {{{ class RuleOp (base class for the rule-ops to come)
class RuleOp(BaseMacro):
    pass
# }}}

def getRule(cls):
    return GetRuleClass(cls.__name__)

def makeSyntaxlang(parent, module):
    usyntaxlang = Language('syntax', parent, module)
    syntaxlang = usyntaxlang.__impl__

    # {{{ SeqOp (continuation op)
    class SeqOp(RuleOp):
        __syntax__ = OperatorSyntax('seq', 100, (ExpressionRule(),
                                                 Rep1(TermRule())))

        macro = staticmethod(lambda *seq:
                             CallFunction(getRule(SequenceRule), *seq))

    syntaxlang.continuationOp = SeqOp
    SeqOp.__language__ = usyntaxlang
    # }}}

    # {{{ lit
    class LiteralOp(RuleOp):

        __syntax__ = OperatorSyntax('lit', 150, (None, ExpressionRule()))
                                    
        macro = staticmethod(lambda lit: CallFunction(getRule(LiteralRule), lit))
    syntaxlang.addOp(LiteralOp)
    # }}}

    syntaxlang.setTokenMacro("string", lambda s: LiteralOp(s))

    # {{{ ' and "
    class SingleQuoteOp(RuleOp):

        __syntax__ = OperatorSyntax(
            "'", 0,
            (None, SequenceRule(FreetextRule(r"[^'\\]*(?:\\.[^'\\]*)*", upto=False),
                                LiteralRule("'"))))

        def macro(text):
            return CallFunction(getRule(LiteralRule), eval(repr(text)))
        macro = staticmethod(macro)

    class DoubleQuoteOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '"', 0,
            (None, SequenceRule(FreetextRule(r'[^"\\]*(?:\\.[^"\\]*)*', upto=False),
                                LiteralRule('"'))))

        def macro(text):
            return CallFunction(getRule(LiteralRule), eval(repr(text)))
        macro = staticmethod(macro)
        
    syntaxlang.addOp(SingleQuoteOp)
    syntaxlang.addOp(DoubleQuoteOp)
    # }}}

    # {{{ expr
    class ExprOp(RuleOp):
        __syntax__ = OperatorSyntax(
            'expr', 200,
            (None, Opt(SequenceRule(LiteralRule("@"),
                                    ChoiceRule(LiteralRule("^"),
                                               TermRule())))))

        macro = staticmethod(lambda lang=None:
                             CallFunction(getRule(ExpressionRule), lang))

    syntaxlang.addOp(ExprOp)
    # }}}

    # {{{ term
    class TermOp(RuleOp):
        __syntax__ = OperatorSyntax(
            'term', 200,
            (None, Opt(SequenceRule(LiteralRule("@"),
                                    ChoiceRule(LiteralRule("^"),
                                               TermRule())))))
                                    

        macro = staticmethod(lambda lang=None:
                             CallFunction(getRule(TermRule), lang))
                    
    syntaxlang.addOp(TermOp)
    # }}}

    # {{{ token
    class TokenOp(RuleOp):
        __syntax__ = OperatorSyntax(
            'token', 200,
            (None, Opt(SequenceRule(LiteralRule("@"),
                                    ChoiceRule(LiteralRule("^"),
                                               TermRule())))))
        
        macro = staticmethod(lambda lang=None:
                             CallFunction(getRule(TokenRule), lang))

    syntaxlang.addOp(TokenOp)
    # }}}

    # {{{ block
    class BlockOp(RuleOp):
        __syntax__ = OperatorSyntax(
            'block', 200,
            (None, Opt(SequenceRule(LiteralRule("@"),
                                    ChoiceRule(LiteralRule("^"),
                                               TermRule())))))

        macro = staticmethod(lambda lang=None:
                             CallFunction(getRule(BlockRule), lang))

    syntaxlang.addOp(BlockOp)
    # }}}

    # {{{ symbol
    class SymbolOp(RuleOp):
        __syntax__ = OperatorSyntax('symbol', 200, (None, None))

        macro = staticmethod(lambda lang=None: CallFunction(getRule(SymbolRule)))
                    
    syntaxlang.addOp(SymbolOp)
    # }}}

    # {{{ eol
    class EolOp(RuleOp):

        __syntax__ = OperatorSyntax('eol', 0, (None, None))

        macro = staticmethod(lambda : CallFunction(getRule(EolRule)))

    syntaxlang.addOp(EolOp)
    # }}}

    # {{{ debug
    class DebugOp(RuleOp):

        __syntax__ = OperatorSyntax('debug', 0,
                                    (None, SequenceRule(LiteralRule("("),
                                                        Opt(ExpressionRule()),
                                                        LiteralRule(")"))))

        def macro(valx=None):
            if isinstance(valx, Symbol):
                valx2 = str(valx)
            else:
                valx2 = valx
            return CallFunction(getRule(DebugRule), valx2)
            
        macro = staticmethod(macro)

    syntaxlang.addOp(DebugOp)
    # }}}

    # {{{ +
    class Rep1Op(RuleOp):
        
        __syntax__ = OperatorSyntax('+', 120, (ExpressionRule(), None))

        macro = staticmethod(lambda rule: CallFunction(getRule(Rep1), rule))

    syntaxlang.addOp(Rep1Op)
    # }}}

    # {{{ *
    class RepOp(RuleOp):
        
        __syntax__ = OperatorSyntax('*', 120, (ExpressionRule(), None))

        macro = staticmethod(lambda rule: CallFunction(getRule(Rep), rule))
        
    syntaxlang.addOp(RepOp)
    # }}}

    # {{{ $
    class NamedRuleOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '$', 110,
            (None, SequenceRule(Opt(TermRule(), None),
                                LiteralRule(':'),
                                ExpressionRule())))

        def macro(namex, rulex):
            if isinstance(namex, Symbol):
                n = str(namex)
            else:
                n = namex

            if isinstance(rulex, OptOp) and len(rulex) == 1:
                subrule = rulex[0]
                return CallFunction(getRule(NamedRule), n, OptOp(subrule, '-'))
            else:
                return CallFunction(getRule(NamedRule), n, rulex)
        macro = staticmethod(macro)

    syntaxlang.addOp(NamedRuleOp)
    # }}}

    # {{{ |
    class ChoiceOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '|', 90, (ExpressionRule(),
                      SequenceRule(ExpressionRule(),
                                   Rep(SequenceRule(LiteralRule("|"),
                                                    ExpressionRule())))))

        macro = staticmethod(lambda *choices:
                             CallFunction(getRule(ChoiceRule), *choices))
        
    syntaxlang.addOp(ChoiceOp)
    # }}}

    # {{{ (
    class ParenOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '(', 0,
            (None, SequenceRule(ExpressionRule(),LiteralRule(')'))))

        macro = staticmethod(lambda rulex: rulex)


    syntaxlang.addOp(ParenOp)
    # }}}

    # {{{ [
    Seq = SequenceRule

    class OptOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '[', 200,
            (None, Seq(ExpressionRule(),
                       LiteralRule(']'),
                       Opt(Seq(LiteralRule("/"),
                               ChoiceRule(LiteralRule("-"),
                                          ExpressionRule()))))))

        def macro(rulex, altx=None):
            if altx == '-':
                return CallFunction(getRule(Opt), rulex)
            else:
                if isinstance(altx, Symbol):
                    altx = str(altx)
                return CallFunction(getRule(Opt), rulex, altx)
        macro = staticmethod(macro)
            
    syntaxlang.addOp(OptOp)
    # }}}

    # {{{ <
    class TrivialOp(RuleOp):

        __syntax__ = OperatorSyntax(
            '<', 0,
            (None, SequenceRule(Opt(ExpressionRule()),
                                LiteralRule('>'))))

        def macro(resultx=None):
            if isinstance(resultx, Symbol):
                resultx = str(resultx)

            if resultx is None:
                return CallFunction(getRule(TrivialRule))
            else:
                return CallFunction(getRule(TrivialRule), resultx)
        macro = staticmethod(macro)
        
    syntaxlang.addOp(TrivialOp)
    # }}}
    
    # {{{ freetext
    class FreetextOp(RuleOp):

        __syntax__ = OperatorSyntax(
            'freetext', 200,
            (None, SequenceRule(Opt(LiteralRule("upto"), False),
                                LiteralRule('/'),
                                FreetextRule(r"[^/\\]*(?:\\.[^/\\]*)*", upto=False),
                                LiteralRule('/'))))

        def macro(upto, terminator):
            return CallFunction(getRule(FreetextRule),
                                terminator,
                                bool(upto))
        macro = staticmethod(macro)
        
    syntaxlang.addOp(FreetextOp)
    # }}}
    
    # {{{ optext
    class OptextOp(RuleOp):

        __syntax__ = OperatorSyntax(
            'optext', 200,
            (None, SequenceRule(NamedRule("lang",
                                          Opt(SequenceRule(LiteralRule("@"),
                                                           SymbolRule()))),
                                LiteralRule('/'),
                                FreetextRule(r"[^/\\]*(?:\\.[^/\\]*)*", upto=False),
                                LiteralRule('/'))))

        def macro(terminator, lang=None):
            return CallFunction(getRule(OptextRule), terminator, lang)
        macro = staticmethod(macro)
        
    syntaxlang.addOp(OptextOp)
    # }}}

    # {{{ symbol:
    class ParsedNameRuleOp(RuleOp):

        __syntax__ = OperatorSyntax('symbol:', 110, (None, ExpressionRule()))

        def macro(rulex):
            return CallFunction(getRule(ParsedNameRule), rulex)
        macro = staticmethod(macro)

    syntaxlang.addOp(ParsedNameRuleOp)
    # }}}

    # make these op-classes available as module fields (needed for quoting)
    globals().update(dict([(cls.__name__, cls)
                           for cls in syntaxlang.operators.values()]))
    globals()['LiteralOp'] = LiteralOp
    globals()['SeqOp'] = SeqOp
    return usyntaxlang
# }}}

# {{{ def rule(syntaxlang, syntax):
def rule(syntaxlang, syntax, env=None):
    env = env or {}
    return language.eval(syntaxlang.__impl__.parse(syntax)[0], env)
# }}}

# {{{ def defopSyntax(syntaxlang):
def defopSyntax(syntaxlang):
    defopsyn = """
    $assoc:['l' <left> | 'r' <right>] 
      $smartspace:['smartspace']
      $binding:term
      $ruledef:expr@syntaxlang
      $imp:[ $kind:('macro' <m> | 'func' <f>)
             $args:[ ($:(symbol ["=" term]/-))*
                     [ "*" $star:symbol ["**" $dstar:symbol]/-
                     | "**" $dstar:symbol
                     ]/-
                   ]/-
             ':'
             $body:block ]
      """

    rrule = rule(syntaxlang, defopsyn,
                 dict(syntaxlang=syntaxlang, star='*', dstar='**'))
    
    return OperatorSyntax('defop', 0, (None, rrule))
# }}}

# {{{ def makeLangLang(syntaxlang, parent, module):
def makeLangLang(syntaxlang, parent, module):
    ulang = Language('langlang', parent, module)
    lang = ulang.__impl__
    
    defop.__syntax__   = defopSyntax(syntaxlang)

    deflang.__syntax__ = OperatorSyntax("deflang", 0, (None,DeflangRule()))

    setlang.__syntax__ = OperatorSyntax("setlang", 0, (None, ExpressionRule()))

    switchlang.__syntax__ = OperatorSyntax(
        "(:", 0,
        (None, SequenceRule(SwitchlangRule(), LiteralRule(")"))))

    outerlang.__syntax__ = OperatorSyntax(
        "(^", 0,
        (None, SequenceRule(ExpressionRule("^"), LiteralRule(")"))))

    getops.__syntax__ = OperatorSyntax("getops", 0, (None, GetopsRule()))
    
    lang.addOp(defop)
    lang.addOp(getops)
    lang.addOp(deflang)
    lang.addOp(setlang)
    lang.addOp(switchlang)
    lang.addOp(outerlang)

    return ulang
# }}}

# {{{ def makeLanguages(homedir, moduledict):
def makeLanguages(logixModuleName, homedir, moduledict):
    import pycompile

    pycompile.installRootopCompilers()
    
    modname = logixModuleName + ".base"

    quotelang = makeQuotelang(parent=None, module=modname)
    syntaxlang = makeSyntaxlang(parent=quotelang, module=modname)
    langlang = makeLangLang(syntaxlang, parent=quotelang, module=modname)
    syntaxlang.__impl__.addOp(langlang.__impl__.getOp("(^"))

    global defaultBaseLang
    defaultBaseLang = langlang

    baselang = Language("base", langlang, modname)
    baselang.operatorBase = PyOp

    filename = homedir + "/base.lx"
    env = dict(__name__=modname,
               base=baselang)
    baselang.__impl__.parse(file(filename, 'U'), mode='exec', execenv=env)
    pycompile.installPyCompilers(baselang.__impl__)

    # HACK: Fix up base lmodule
    mod = new.module(filename)
    mod.__file__ = filename
    vars(mod).update(env)
    moduledict[modname] = mod
    mod.langlang = langlang
    mod.syntaxlang = syntaxlang
    mod.quotelang = quotelang

    return quotelang, syntaxlang, langlang, baselang
# }}}

# {{{ def makeQuotelang(parent, module):
def makeQuotelang(parent, module):
    ulang = Language("quotelang", parent, module)
    lang = ulang.__impl__

    quote.__syntax__ = OperatorSyntax("`", 0, (None, ExpressionRule()))

    escape.__syntax__ = OperatorSyntax("\\", 200, (None, SequenceRule(
        NamedRule('extra', Rep(LiteralRule("\\"))),
        ChoiceRule(NamedRule('localmodule', LiteralRule("@")),
                   SequenceRule(NamedRule('splice', Opt(LiteralRule('*'))),
                                ChoiceRule(SequenceRule(LiteralRule("("),
                                                        ExpressionRule(),
                                                        LiteralRule(")")),
                                           ExpressionRule()))))))

    opquote.__syntax__ = OperatorSyntax("``", 0, (None, TokenRule()))

    lang.addOp(quote)
    lang.addOp(escape)
    lang.addOp(opquote)

    return ulang
# }}}
