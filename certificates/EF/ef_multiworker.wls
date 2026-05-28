#!/usr/bin/env wolframscript

ClearAll["Global`*"];
$Assumptions = {0 < \[Mu] < L, L > 0, 0 < \[Epsilon] <= 1,
     Element[{L, \[Mu], \[Epsilon]}, Reals]};

(* Default n=2 if not provided *)
If[ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] >= 2 && 
   StringMatchQ[$ScriptCommandLine[[2]], NumberString],
    n = ToExpression[$ScriptCommandLine[[2]]],
    n = 2
];

Print["Building SOS certificate for n = ", n];

(* --- definitions --- *)

(* Variables *)
g0 = Array[gk, n];     (* gradients g_k^(i) *)
e0 = Array[ek, n];     (* error e_k^(i) *)
cV = Array[EC, n];    (* E[m_i] where m_i = C(e0[[i]] + eta g0[[i]]) *)
c2V = Array[EC2, n];  (* E[||m_i||^2] *)
f0 = Array[fk, n];     (* function values f_i(x_k) *)
fs = Array[fstar, n]; (* function values f_i(x_star) *)
(* Note: xk and xs are symbols defined implicitly by usage *)

(* Parameters---note that the rate is encoded differently for performance reasons *)
\[Eta] = 2/(L + \[Mu]) * (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]);
\[Lambda] = \[Eta]/(L + \[Mu]) * ((1 - Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) * Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);
\[Rho] = Sqrt[\[Epsilon]] + 1/4 * (1 + Sqrt[\[Epsilon]]) * (L - \[Mu]) * \[Lambda];
\[Nu] = 1/Sqrt[\[Epsilon]];


(* --- Inequalities --- *)

(* 1. Smooth/strong-convexity interpolation constraints; summed in diff below *)
(* I1: f(x_k) - f(x_star) - <g_k, x_k - x_star> + ... <= 0 *)
I1[i_] := f0[[i]] - fs[[i]] - (xk - xs) * g0[[i]] + (1/(2*L)) * g0[[i]]^2 + (\[Mu]/(2*(1 - \[Mu]/L))) * ((xk - xs) - g0[[i]]/L)^2;

(* I2: f(x_star) - f(x_k) + ... <= 0 *)
(* Uses interpolation point (x_star, 0, f_i(x_star)); assumes each f_i is stationary at the common x_star *)
I2[i_] := fs[[i]] - f0[[i]] + (1/(2*L)) * g0[[i]]^2 + (\[Mu]/(2*(1 - \[Mu]/L))) * ((xk - xs) - g0[[i]]/L)^2;

(* 2. Compression Inequality (Summed over i) *)
(* I_comp^(i): (1-eps)||v||^2 - 2<v, E[C(v)]> + E[||C(v)||^2] <= 0 *)
CompIneq[i_] := (1 - \[Epsilon]) * (e0[[i]] + \[Eta] * g0[[i]])^2 - 2 * (e0[[i]] + \[Eta] * g0[[i]]) * cV[[i]] + c2V[[i]];


(* --- Lyapunov Function V0 --- *)
(* V0 = (1/(n sqrt(eps))) ||sum e_i||^2 + sum ||(xk - xs) - e_i||^2 *)
V0 = (1/(n * Sqrt[\[Epsilon]])) * (Sum[e0[[i]], {i, 1, n}])^2 +
  Sum[((xk - xs) - e0[[i]])^2, {i, 1, n}];


(* --- Lyapunov Function V1 (Expectation) --- *)
(* Mirror V0 with e1^(i) = v_i - m_i and x_{k+1} = x_k - (1/n) Sum m_i. *)
v[i_] := e0[[i]] + \[Eta] * g0[[i]];
vbar = Sum[v[i], {i, 1, n}]/n;
cbar = Sum[cV[[i]], {i, 1, n}]/n;
mVar[i_] := c2V[[i]] - cV[[i]]^2;
mVarBar = (1/n^2) * Sum[mVar[j], {j, 1, n}];

(* Assuming independent worker compression randomness:
   E[||sum_i (v_i - m_i)||^2] = ||sum_i (v_i - E[m_i])||^2 + sum_i Var(m_i). *)
V1 = (1/(n * Sqrt[\[Epsilon]])) * (
    (Sum[v[i] - cV[[i]], {i, 1, n}])^2 +
    Sum[mVar[i], {i, 1, n}]
  ) +
  Sum[
    ((xk - xs) - v[i] + (cV[[i]] - cbar))^2 +
    (1 - 2/n) * mVar[i] + mVarBar,
    {i, 1, n}
  ];


(* --- Residual S --- *)
a = (\[Rho] - Sqrt[\[Epsilon]]) (1 + Sqrt[\[Epsilon]])/Sqrt[\[Epsilon]];


(* Mean term *)
e0bar = Sum[e0[[i]], {i, 1, n}]/n;
Smean = (e0bar - (\[Rho] - 1)/a * (xk - xs) + 2 (Sqrt[\[Epsilon]] - 1)/(a (L + \[Mu])) * g0bar)^2;

(* Variance term *)
g0bar = Sum[g0[[i]], {i, 1, n}]/n;
Svar = Sum[(g0[[i]] - g0bar)^2, {i, 1, n}];

(* Mixed term *)
Smix = Sum[ ((e0[[i]] - e0bar) - (Sqrt[\[Epsilon]] * \[Eta])/(\[Rho] - Sqrt[\[Epsilon]]) * (g0[[i]] - g0bar))^2, {i, 1, n} ];

(* Mixed Compression term *)
(* SmixComp = Sum_i E[||(v_i - vbar) - (m_i - mbar)||^2], expanded under independent worker compression randomness *)
SmixComp = Sum[
   (v[i] - vbar - (cV[[i]] - cbar))^2 + (1 - 2/n) * mVar[i] + mVarBar,
   {i, 1, n}
];

c = \[Lambda]/(L - \[Mu]) - (Sqrt[\[Epsilon]] * \[Rho] * \[Eta]^2)/(\[Rho] - Sqrt[\[Epsilon]]);

S = n * a * Smean + c * Svar + (\[Rho] - Sqrt[\[Epsilon]]) * Smix + (\[Nu] - 1) * SmixComp;

(* --- Final Difference --- *)
diff = \[Rho] * V0 +
   \[Lambda] * Sum[I1[i] + I2[i], {i, 1, n}] +
   \[Nu] * Sum[CompIneq[i], {i, 1, n}] -
   S - V1;

(* --- Verification --- *)
tStart = AbsoluteTime[];

(* Collect variables: include xk and xs *)
allVars = Join[{xk, xs}, Flatten[g0], Flatten[e0], Flatten[cV], Flatten[c2V], Flatten[f0], Flatten[fs]];
coeffs = Flatten[Normal[CoefficientArrays[diff, allVars]]];

(* Simplify *)
zeros = (Simplify[# == 0, Assumptions -> $Assumptions] & /@ coeffs);
result = And @@ TrueQ /@ zeros;

tEnd = AbsoluteTime[];

Print["Certification completed in ", ToString[NumberForm[tEnd - tStart, {Infinity, 2}], OutputForm], " seconds."];

If[result,
  Print["\nVerification Result: \:2705"],
  Print["\nVerification Result: \:274c"]
];
