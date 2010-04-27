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
from os import path
import cPickle


from parser import *
from language import Language
import language
import data
from data import Symbol, Doc
# }}}

# {{{ Base Operators

base_ns = "base"

std_ns = "std"

callOp = Symbol("lx", "call")
listOp = Symbol("lx", "list")
tupleOp = Symbol("lx", "tuple")
dictOp = Symbol("lx", "dict")
getlmoduleOp = Symbol("lx", "getlmodule")
subscriptOp = Symbol("lx", "subscript")
sliceOp = Symbol("lx", "slice")

langlang_ns = "langlang"

switchlang = Symbol(langlang_ns, "(:")
outerlang  = Symbol(langlang_ns, "(^")
defop  = Symbol(langlang_ns, "defop")
deflang  = Symbol(langlang_ns, "deflang")
setlang  = Symbol(langlang_ns, "setlang")
getops  = Symbol(langlang_ns, "getops")

syntaxlang_ns = "syntax"

getRuleClass = Symbol(syntaxlang_ns, "get-rule")


quotelang_ns = "quotelang"

quote = Symbol(quotelang_ns, "`")
escape = Symbol(quotelang_ns, "\\")

# }}}

# {{{ syntax lang

def makeSyntaxlang(parent, module):
    op = Doc.new

    Seq = SequenceRule
    
    def makeRule(*args, **kws):
        className = args[0].__name__
        args = args[1:]
        return op(callOp, op(getRuleClass, className), *args, **kws)
    
    ulang = Language('syntax', parent, module)
    lang = ulang.__impl__

    # {{{ seq (continuation op)
    seq = lang.newOp('seq', 100, (ExpressionRule(), Rep1(TermRule())))
    seq.macro = lambda *seq: makeRule(SequenceRule, *seq)
    lang.continuationOp = seq
    # }}}

    # {{{ lit
    lang.newOp('lit', 150, (None, ExpressionRule())
               
               ).macro = lambda lit: makeRule(LiteralRule, lit)
    # }}}

    # {{{ ' and "
    lang.newOp("'", 0,
               (None, SequenceRule(FreetextRule(r"[^'\\]*(?:\\.[^'\\]*)*",
                                                upto=False),
                                   LiteralRule("'")))
               ).macro = lambda text: makeRule(LiteralRule, eval(repr(text)))

    lang.newOp('"', 0,
               (None, SequenceRule(FreetextRule(r'[^"\\]*(?:\\.[^"\\]*)*',
                                                upto=False),
                                   LiteralRule('"')))
               ).macro = lambda text: makeRule(LiteralRule, eval(repr(text)))
    # }}}

    # {{{ expr
    lang.newOp('expr', 200,
               (None, Opt(SequenceRule(LiteralRule("@"),
                                       ChoiceRule(LiteralRule("^"),
                                                  TermRule()))))

               ).macro = lambda lang=None: makeRule(ExpressionRule, lang)
    # }}}

    # {{{ term
    lang.newOp('term', 200,
               (None, Opt(SequenceRule(LiteralRule("@"),
                                       ChoiceRule(LiteralRule("^"),
                                                  TermRule()))))
               
               ).macro = lambda lang=None: makeRule(TermRule, lang)
    # }}}

    # {{{ block
    def block_macro(kind=None, x=None):
        if kind == "lang":
            return makeRule(BlockRule, x)
        elif kind == "rule":
            return makeRule(BlockRule, None, x)
        else:
            return makeRule(BlockRule)

    
    lang.newOp('block', 200,
               (None, Opt(ChoiceRule(SequenceRule(TrivialRule("lang"),
                                                  LiteralRule("@"),
                                                  ChoiceRule(LiteralRule("^"),
                                                             TermRule())),
                                     SequenceRule(TrivialRule("rule"),
                                                  LiteralRule(":"),
                                                  ExpressionRule()))))

               ).macro = block_macro
    # }}}

    # {{{ symbol
    lang.newOp('symbol', 200, (None, None)

               ).macro = lambda: makeRule(SymbolRule)
    # }}}

    # {{{ eol
    lang.newOp('eol', 0, (None, None)

               ).macro = lambda : makeRule(EolRule)
    # }}}

    # {{{ debug
    def debug_macro(valx=None):
        if isinstance(valx, Symbol):
            valx2 = str(valx)
        else:
            valx2 = valx
        return makeRule(DebugRule, valx2)
    
    lang.newOp('debug', 0,
               (None, SequenceRule(LiteralRule("("),
                                   Opt(ExpressionRule()),
                                   LiteralRule(")")))
               
               ).macro = debug_macro
    # }}}

    # {{{ +
    lang.newOp('+', 110, (ExpressionRule(), None)

               ).macro = lambda rule:makeRule(Rep1, rule)
    # }}}

    # {{{ *
    lang.newOp('*', 110, (ExpressionRule(), None)

               ).macro = lambda rule:makeRule(Rep, rule)
    # }}}

    # {{{ $
    def dollar_macro(namex, rulex):
        if namex == data.none:
            n = None
        elif isinstance(namex, Symbol):
            n = str(namex)
        else:
            n = namex

        optOp = Symbol(syntaxlang_ns, "[")
        if isDoc(rulex, optOp) and rulex.contentLen() == 1:
            # The rule being named is optional ([...]) with no alternative.
            # Set the alternative to 'omit' (because it's named)
            return makeRule(NamedRule, n, op(optOp, rulex[0], '-'))
        else:
            return makeRule(NamedRule, n, rulex)
        
    lang.newOp('$', 120,
               (None, SequenceRule(Opt(TermRule(), None),
                                   LiteralRule(':'),
                                   ExpressionRule()))

               ).macro = dollar_macro
    # }}}

    # {{{ |
    lang.newOp('|', 90, (ExpressionRule(),
                         SequenceRule(ExpressionRule(),
                                      Rep(SequenceRule(LiteralRule("|"),
                                                       ExpressionRule()))))

               ).macro = lambda *choices: makeRule(ChoiceRule, *choices)
    # }}}

    # {{{ (
    lang.newOp('(', 0,
               (None, SequenceRule(ExpressionRule(),LiteralRule(')')))

               ).macro = lambda rulex: rulex
    # }}}

    # {{{ [
    def opt_macro(rulex, altx=None):
        if altx == '-':
            return makeRule(Opt, rulex)
        else:
            if isinstance(altx, Symbol):
                altx = str(altx)
            return makeRule(Opt, rulex, altx)

    lang.newOp('[', 200,
               (None, Seq(ExpressionRule(),
                          LiteralRule(']'),
                          Opt(Seq(LiteralRule("/"),
                                  ChoiceRule(LiteralRule("-"),
                                             ExpressionRule())))))

               ).macro = opt_macro
    # }}}

    # {{{ <
    def trivial_macro(resultx=None):
        if isinstance(resultx, Symbol):
            resultx = str(resultx)
            
        if resultx is None:
            return makeRule(TrivialRule)
        else:
            return makeRule(TrivialRule, resultx)
        
    lang.newOp('<', 0,
               (None, SequenceRule(Opt(ExpressionRule()),
                                   LiteralRule('>')))

               ).macro = trivial_macro
    # }}}
    
    # {{{ freetext
    lang.newOp('freetext', 200,
               (None, SequenceRule(Opt(LiteralRule("upto"), False),
                                   LiteralRule('/'),
                                   FreetextRule(r"[^/\\]*(?:\\.[^/\\]*)*", upto=False),
                                   LiteralRule('/')))
               
               ).macro = lambda upto, terminator: makeRule(FreetextRule,
                                                           terminator,
                                                           bool(upto))
    # }}}
    
    # {{{ freetoken
    lang.newOp('freetoken', 200,
               (None, SequenceRule(LiteralRule('/'),
                                   FreetextRule(r"[^/\\]*(?:\\.[^/\\]*)*", upto=False),
                                   LiteralRule('/')))
               
               ).macro = lambda terminator: makeRule(FreetokenRule, terminator)
    # }}}
    
    # {{{ optext
    lang.newOp('optext', 200,
               (None, SequenceRule(NamedRule("lang", Opt(SymbolRule(), None)),
                                   NamedRule("ops", Rep(SequenceRule(LiteralRule('"'),
                                                                     FreetextRule(r'[^\\"]*(?:\\.[^\\"]*)*', upto=False),
                                                                     LiteralRule('"')))),
                                   Opt(LiteralRule("oneline"), False),
                                   LiteralRule('/'),
                                   FreetextRule(r"[^/\\]*(?:\\.[^/\\]*)*", upto=False),
                                   LiteralRule('/')))

               ).macro = lambda oneline, terminator, lang, ops: makeRule(OptextRule,
                                                                         lang,
                                                                         Doc(listOp, [eval('"%s"' % op) for op in ops]),
                                                                         bool(oneline),
                                                                         terminator)
    # }}}

    # {{{ "symbol:" 
    lang.newOp('symbol:', 110, (None, ExpressionRule())

               ).macro = lambda rulex: makeRule(ParsedNameRule, rulex)
    # }}}

    return ulang
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
    ulang = Language(langlang_ns, parent, module)
    lang = ulang.__impl__
    
    lang.newOpFromSyntax(defopSyntax(syntaxlang))

    lang.newOp("deflang", 0, (None,DeflangRule()))

    lang.newOp("setlang", 0, (None, ExpressionRule()))

    lang.newOp("(:", 0, (None,
                         SequenceRule(SwitchlangRule(),
                                      LiteralRule(")")))

               ).macro = lambda x: x

    lang.newOp("(^", 0, (None,
                         SequenceRule(ExpressionRule("^"),
                                      LiteralRule(")")))

               ).macro = lambda x: x

    Seq = SequenceRule
    lang.newOp("getops", 0, (None, Seq(ExpressionRule(),
                                       Opt(Seq(LiteralRule(","),
                                               Rep1(FreetextRule(r"\s*\S+",
                                                                 False)))))))

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

    # {{{ baselang = create baselang (possibly from cache)
    baselang = Language("base", langlang, modname)

    filename = homedir + "/base.lx"
    env = dict(__name__=modname,
               base=baselang,
               langlang=langlang)

    def persistent_load(pid):
        res = {'nothing': nothing,
               'eol': Tokenizer.endOfLine,
               'eob': Tokenizer.endOfBlock,
               'eof': Tokenizer.endOfFile }.get(pid)
        if res: return res
        else: raise cPickle.UnpicklingError, 'Invalid persistent id'

    def persistent_id(x):
        return {id(nothing): "nothing",
                id(Tokenizer.endOfLine) : "eol",
                id(Tokenizer.endOfBlock): "eob",
                id(Tokenizer.endOfFile):  "eof"}.get(id(x))
  
    opFile = "%s/logix_opcache" % homedir
  
    if not path.exists(opFile):
        baselang.__impl__.parse(file(filename, 'U'),
                                mode='exec', execenv=env)
        oplst = [x.syntax for x in baselang.__impl__.operators.values()]

        p = cPickle.Pickler(file(opFile,"w+"))
        p.persistent_id = persistent_id
        p.dump(oplst)

    else:
        p = cPickle.Unpickler(file(opFile))
        p.persistent_load = persistent_load
        for syntax in p.load():
            baselang.__impl__.newOpFromSyntax(syntax)
    # }}}

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

    lang.newOp("`", 0, (None, ExpressionRule()))

    lang.newOp("\\", 200, (None, SequenceRule(
        NamedRule('extra', Rep(LiteralRule("\\"))),
        ChoiceRule(NamedRule('localmodule', LiteralRule("@")),
                   SequenceRule(NamedRule('splice', Opt(LiteralRule('*'))),
                                ChoiceRule(SequenceRule(LiteralRule("("),
                                                        ExpressionRule(),
                                                        LiteralRule(")")),
                                           ExpressionRule()))))))

    return ulang
# }}}
