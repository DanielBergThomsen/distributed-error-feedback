#!/usr/bin/env wolframscript

ClearAll["Global`*"];
$Assumptions = {0 < \[Mu] < L, L > 0, 0 < \[Epsilon] <= 1,
     Element[{L, \[Mu], \[Epsilon], \[Lambda], \[Rho]}, Reals]};

(* Default n=2 if not provided *)
If[ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] >= 2 && 
   StringMatchQ[$ScriptCommandLine[[2]], NumberString],
    n = ToExpression[$ScriptCommandLine[[2]]],
    n = 2
];

Print["Building SOS certificate for n = ", n];

(* --- definitions --- *)

(* Variables *)
g0 = Array[gk, n];   (* gradients at k *)
g1 = Array[gk1, n];  (* gradients at k+1 *)
d0 = Array[dk, n];   (* state d_k *)
cV = Array[EC, n];   (* E[m_i] where m_i = C(g1[[i]] - d0[[i]]) *)
c2V = Array[EC2, n]; (* E[||m_i||^2] *)
f0 = Array[fk, n];   (* function values at x_k *)
f1 = Array[fk1, n];  (* function values at x_{k+1} *)

(* Parameters *)
\[Eta] = Simplify[2/(L + \[Mu]) * (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]), $Assumptions];

\[Lambda] = Simplify[Sqrt[\[Epsilon]]/(\[Eta] (L + \[Mu])) * 
  ((1 - Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) * 
   Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]), $Assumptions];

\[Psi] = Simplify[
  1 - Sqrt[\[Epsilon]] + Sqrt[(1 + Sqrt[\[Epsilon]])^2 + Sqrt[\[Epsilon]]*16 L \[Mu]/(L - \[Mu])^2],
  $Assumptions
];
\[Rho] = Simplify[
  Sqrt[\[Epsilon]] + ((1 - Sqrt[\[Epsilon]])/2) * ((L - \[Mu])/(L + \[Mu]))^2 * \[Psi],
  $Assumptions
];

(* Smooth/strong-convexity interpolation inequalities between x_k and x_{k+1} *)
xStep = -\[Eta] * (1/n) * Sum[d0[[j]], {j, 1, n}];
I1[i_] := f0[[i]] - f1[[i]] + g0[[i]] * xStep + (g1[[i]] - g0[[i]])^2/(2 L) +
  (\[Mu]/(2 (1 - \[Mu]/L))) * (xStep - (g1[[i]] - g0[[i]])/L)^2;
I2[i_] := f1[[i]] - f0[[i]] - g1[[i]] * xStep + (g1[[i]] - g0[[i]])^2/(2 L) +
  (\[Mu]/(2 (1 - \[Mu]/L))) * (xStep - (g1[[i]] - g0[[i]])/L)^2;

(* Compression Inequality Term *)
CompIneq[i_] := (1 - \[Epsilon]) (g1[[i]] - d0[[i]])^2 - 2 (g1[[i]] - d0[[i]]) cV[[i]] + c2V[[i]];

(* Lyapunov Function V0 *)
V0 = (Sqrt[\[Epsilon]]/n) * (Sum[g0[[i]], {i, 1, n}])^2 +
  Sum[(g0[[i]] - d0[[i]])^2, {i, 1, n}];


(* --- construction --- *)

(* V1 = (sqrt(eps)/n) ||sum g1||^2 + sum E[||g1 - d0 - m||^2]. *)
V1 = (Sqrt[\[Epsilon]]/n) * (Sum[g1[[i]], {i, 1, n}])^2 +
  Sum[(g1[[i]] - d0[[i]])^2 - 2 (g1[[i]] - d0[[i]]) cV[[i]] + c2V[[i]], {i, 1, n}];

(* 3. Residual S *)
(* Coefficients pre-simplified *)
a = Simplify[\[Rho] - \[Epsilon] + \[Lambda] * \[Eta]^2 * (L * \[Mu])/(L - \[Mu]), $Assumptions];
b = Simplify[(\[Eta] * \[Lambda])/2 * (L + \[Mu])/(L - \[Mu]), $Assumptions];
c = Simplify[\[Lambda]/(L - \[Mu]) - (\[Epsilon] * \[Rho])/(\[Rho] - \[Epsilon]), $Assumptions];

g0bar = (1/n) * Sum[g0[[i]], {i, 1, n}];
g1bar = (1/n) * Sum[g1[[i]], {i, 1, n}];
d0bar = (1/n) * Sum[d0[[i]], {i, 1, n}];

Smean = (d0bar + (1/a) * ( (\[Epsilon] + b) * g1bar - (\[Rho] + b) * g0bar ))^2;
Svar = Sum[ ( (g1[[i]] - g1bar) - (g0[[i]] - g0bar) )^2, {i, 1, n} ];
Smix = Sum[ ( (d0[[i]] - d0bar) + (\[Epsilon]/(\[Rho] - \[Epsilon])) * (g1[[i]] - g1bar) - (\[Rho]/(\[Rho] - \[Epsilon])) * (g0[[i]] - g0bar) )^2, {i, 1, n} ];
S = n * a * Smean + c * Svar + (\[Rho] - \[Epsilon]) * Smix;

(* 4. Final Difference *)
(* Algebraic identity:
   \[Lambda] Sum[I1 + I2] + Sum[CompIneq] = V1 - \[Rho] V0 + S
*)
diff = \[Lambda] * Sum[I1[i] + I2[i], {i, 1, n}] +
  Sum[CompIneq[i], {i, 1, n}] - V1 + \[Rho] * V0 - S;


(* --- verification --- *)

tStart = AbsoluteTime[];

(* Check if the polynomial identity is zero by inspecting coefficients *)
allVars = Join[Flatten[g0], Flatten[g1], Flatten[d0], Flatten[cV], Flatten[c2V], Flatten[f0], Flatten[f1]];
coeffs = Flatten[Normal[CoefficientArrays[diff, allVars]]];

(* Simplify each coefficient to check if it is zero *)
zeros = (Simplify[# == 0, Assumptions -> $Assumptions] & /@ coeffs);
result = And @@ TrueQ /@ zeros;

tEnd = AbsoluteTime[];

Print["Certification completed in ", ToString[NumberForm[tEnd - tStart, {Infinity, 2}], OutputForm], " seconds."];

If[result,
  Print["\nVerification Result: \:2705"], (* Check Mark Button Emoji *)
  Print["\nVerification Result: \:274c"]  (* Cross Mark Emoji *)
];
