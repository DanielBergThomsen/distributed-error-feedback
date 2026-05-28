ClearAll["Global`*"];

$Assumptions = {
  0 < \[Mu] < L, L > 0, 0 < \[Epsilon] <= 1,
  Element[{L, \[Mu], \[Epsilon]}, Reals]
};

eta = 2/(L + \[Mu]) * (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]);

lam = Sqrt[\[Epsilon]]/((eta (L + \[Mu]))) ((1 -
        Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 +
        Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 +
        16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);

rhoClosed = Sqrt[\[Epsilon]] + ((1 - Sqrt[\[Epsilon]])/2) *
    ((L - \[Mu])/(L + \[Mu]))^2 * (1 - Sqrt[\[Epsilon]] +
      Sqrt[(1 + Sqrt[\[Epsilon]])^2 +
        Sqrt[\[Epsilon]]*16 L \[Mu]/(L - \[Mu])^2]);

\[Eta] = eta;
\[Lambda] = lam;
\[Rho] = rhoClosed;

a = \[Rho] - \[Epsilon] + \[Lambda] \[Eta]^2 L \[Mu]/(L - \[Mu]);
c = \[Lambda]/(L - \[Mu]) - \[Epsilon] \[Rho]/(\[Rho] - \[Epsilon]);

check[label_, expr_] := Print[label, " ", If[TrueQ[expr], "PASS", "FAIL"]];
t = Sqrt[(1 + Sqrt[\[Epsilon]])^2 + (16 L \[Mu] Sqrt[\[Epsilon]])/(L - \[Mu])^2];

rhoMinusEps = Simplify[\[Rho] - \[Epsilon], Assumptions -> $Assumptions];
rhoMinusEpsSplit = (Sqrt[\[Epsilon]] - \[Epsilon]) +
   ((1 - Sqrt[\[Epsilon]])/2) * ((L - \[Mu])/(L + \[Mu]))^2 * (1 - Sqrt[\[Epsilon]] + t);

check["rho-eps rewrite valid?",
  Simplify[rhoMinusEps == rhoMinusEpsSplit, Assumptions -> $Assumptions]];
check["t^2 - (1 + sqrt eps)^2 > 0?",
  Simplify[t^2 - (1 + Sqrt[\[Epsilon]])^2 > 0, Assumptions -> Join[$Assumptions, {\[Epsilon] < 1}]]];

cNum = Simplify[(L - \[Mu]) (\[Rho] - \[Epsilon]) c, Assumptions -> $Assumptions];
check["c reconstruction valid?",
  Simplify[c == cNum/((L - \[Mu]) (\[Rho] - \[Epsilon])), Assumptions -> $Assumptions]];
K = ((L - \[Mu])/(L + \[Mu]))^2;
u = 1 - Sqrt[\[Epsilon]] + t;
s = Sqrt[(L - \[Mu])^2 + (16 L \[Mu] Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]])^2];
check["s relation valid?",
  Simplify[s == (L - \[Mu]) t/(1 + Sqrt[\[Epsilon]]), Assumptions -> $Assumptions]];

lamOverAlt = (Sqrt[\[Epsilon]] (1 + Sqrt[\[Epsilon]])/(2 (1 - Sqrt[\[Epsilon]]))) * u;
check["lambda/(L-mu) rewrite valid?",
  Simplify[\[Lambda]/(L - \[Mu]) == lamOverAlt, Assumptions -> $Assumptions]];

rhoAlt = Sqrt[\[Epsilon]] + ((1 - Sqrt[\[Epsilon]])/2) * K * u;
check["rho rewrite valid?",
  Simplify[\[Rho] == rhoAlt, Assumptions -> $Assumptions]];

cNumAlt = Simplify[lamOverAlt (rhoAlt - \[Epsilon]) - \[Epsilon] rhoAlt, Assumptions -> $Assumptions];
uSym = Unique["u"];
rhoAltU = Sqrt[\[Epsilon]] + ((1 - Sqrt[\[Epsilon]])/2) * K * uSym;
lamOverAltU = (Sqrt[\[Epsilon]] (1 + Sqrt[\[Epsilon]])/(2 (1 - Sqrt[\[Epsilon]]))) * uSym;
cNumU = Simplify[lamOverAltU (rhoAltU - \[Epsilon]) - \[Epsilon] rhoAltU, Assumptions -> $Assumptions];

a2 = Simplify[Coefficient[cNumU, uSym, 2], Assumptions -> $Assumptions];
a1 = Simplify[Coefficient[cNumU, uSym, 1], Assumptions -> $Assumptions];
a0 = Simplify[Coefficient[cNumU, uSym, 0], Assumptions -> $Assumptions];

cNumAt2 = Simplify[cNumU /. uSym -> 2, Assumptions -> $Assumptions];
check["c numerator a2 > 0?", Simplify[a2 > 0, Assumptions -> $Assumptions]];
check["c numerator a1 > 0?", Simplify[a1 > 0, Assumptions -> $Assumptions]];

cNumAt2Alt = (Sqrt[\[Epsilon]]/(L + \[Mu])^2) *
   ((1 + Sqrt[\[Epsilon]] + \[Epsilon]) (L - \[Mu])^2 + 4 Sqrt[\[Epsilon]] L \[Mu]);
check["c numerator at u=2 rewrite valid?",
  Simplify[cNumAt2 == cNumAt2Alt, Assumptions -> $Assumptions]];
check["c numerator at u=2 > 0?",
  Simplify[cNumAt2Alt > 0, Assumptions -> $Assumptions]];

a1p = Simplify[4 a2 + a1, Assumptions -> $Assumptions];
cNumShift = Simplify[a2 (uSym - 2)^2 + a1p (uSym - 2) + cNumAt2, Assumptions -> $Assumptions];
check["c numerator shifted-factor form valid?",
  Simplify[cNumU == cNumShift, Assumptions -> $Assumptions]];
check["c numerator (4 a2 + a1) > 0?", Simplify[a1p > 0, Assumptions -> $Assumptions]];
