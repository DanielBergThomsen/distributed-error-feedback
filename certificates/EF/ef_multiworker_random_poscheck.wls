ClearAll["Global`*"];
$Assumptions = {0 < \[Mu] < L, L > 0, 0 < \[Epsilon] <= 1,
     Element[{L, \[Mu], \[Epsilon]}, Reals]};

s = Sqrt[\[Epsilon]];
t = Sqrt[(L - \[Mu])^2 + 16 L \[Mu] s/(1 + s)^2];

\[Eta] = 2/(L + \[Mu]) (1 - s)/(1 + s);
\[Lambda] = \[Eta]/(L + \[Mu]) ((1 - s) (L - \[Mu]) + (1 + s) t);
\[Rho] = s + 1/4 (1 + s) (L - \[Mu]) \[Lambda];
\[Nu] = 1/s;

(* Positive-form rewrites *)
rhoMinus = (1 - s) (L - \[Mu]) ((1 - s) (L - \[Mu]) + (1 + s) t)/(2 (L + \[Mu])^2);
aPos = (1 + s) rhoMinus/s;
cPos = (2 (1 - s) ((1 - s) (L - \[Mu]) + (1 + s) t))/((1 + s)^2 (L - \[Mu]) (L + \[Mu])^2);

check[label_, expr_] := Print[label, " ", If[TrueQ[expr], "PASS", "FAIL"]];

a = (1 + s) (\[Rho] - s)/s;
cG = \[Lambda]/(L - \[Mu]) - (s \[Rho] \[Eta]^2)/(\[Rho] - s);

check["rho - sqrt eps rewrite valid?",
  Simplify[rhoMinus == \[Rho] - s, Assumptions -> $Assumptions]];
check["rho - sqrt eps > 0?",
  FullSimplify[rhoMinus > 0, Assumptions -> Join[$Assumptions, {\[Epsilon] < 1}]]];

check["a rewrite valid?",
  Simplify[aPos == a, Assumptions -> $Assumptions]];
check["a > 0?",
  FullSimplify[aPos > 0, Assumptions -> Join[$Assumptions, {\[Epsilon] < 1}]]];

check["nu - 1 > 0?",
  FullSimplify[\[Nu] - 1 > 0, Assumptions -> Join[$Assumptions, {\[Epsilon] < 1}]]];

check["c rewrite valid?",
  FullSimplify[cPos == cG, Assumptions -> $Assumptions]];
check["c > 0?",
  FullSimplify[cPos > 0, Assumptions -> Join[$Assumptions, {\[Epsilon] < 1}]]];
