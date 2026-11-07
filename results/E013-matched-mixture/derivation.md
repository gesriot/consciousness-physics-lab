# Matched slab mixtures: definition-only construction

Write s=zT. For a slab of mean xT, the normalized response is
L_x(s)=tanh(sqrt(3xs))/sqrt(3xs). Its expansion is
1-xs+(6/5)x²s²-(51/35)x³s³+O(s⁴). The sphere expansion is
1-s+(10/7)s²-(15/7)s³+O(s⁴). Thus the normalized second and third
moments are12/5 and306/35 for a slab,20/7 and90/7 for a sphere.
The leading slab high-frequency coefficient is1/sqrt(3x), versus
sqrt(3/5) for a sphere. Weighting slab responses preserves positivity.

The four constraints on a two-component mixture are sum(w)=1, sum(wx)=1,
sum(wx²)=25/21 and sum(w/sqrt(x))=3/sqrt(5). Solving these equations
at60 decimal digits gives

- weights:0.18979085320866434855,0.81020914679133565145;
- normalized mean times:0.09826011362575569387,1.21123185671875465486.

All residuals are at most1.6e-61 before decimal output rounding. The two
components share the same target scale T, not the same individual mean time.
A finite modal truncation and a finite-frequency probe need not reproduce these
limiting features exactly; their registered operational allowances are tested
separately, without replacing them by exact theoretical constraints.

For a fifth constraint, matching the third moment requires sum(wx³)=25/17.
Fixing the smallest mean time at0.03 yields a positive three-component solution:

- weights:0.05707829832827197467,0.19571807172477895210,0.74720362994694907323;
- normalized mean times:0.03,0.32345935038042538116,1.25130656923967073494.

The five equation residuals are below4.4e-55 in the60-digit calculation.
This calculation was made before task generation and does not entail an
additional three-slab task experiment.

More generally, finitely many linear features of a mixture correspond to a
finite feature matrix evaluated on its support points. With more support points
than independent constraints, that matrix has a nonzero null vector. Starting
with strictly positive weights, sufficiently small perturbations in either
null-vector direction preserve all features and positivity. Including the
normalization row preserves the total weight. Distinct supports carry linearly
independent slab response functions (their analytic continuations have distinct
pole sets generically), so such perturbations can change the response away from
the checked features. This argument establishes nonidentifiability of the broad
mixture class, not existence of a positive finite realization of every possible
target feature vector. The explicit constructions above establish the two cases
needed here for the sphere.
