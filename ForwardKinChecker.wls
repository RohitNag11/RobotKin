#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Input the DH parameters as a list of lists *)
(* Each sublist should contain the parameters [a, alpha, d, theta] *)
(* For example, here is a sample set of DH parameters for a 3-link manipulator *)
dhParameters = {
  {0, 0, L1, theta1[t]},
  {L2, 0, d, theta2[t]},
  {L3, 0, d3[t], 0},
  {L4, 0, 0, 0}
};

(* Change this value to the desired end link number *)
endLinkNumber = 4;



(* Define a function to compute the DH transformation matrix *)
dhTransform[a_, alpha_, d_, theta_] := {
  {Cos[theta], -Sin[theta], 0, a},
  {Sin[theta]*Cos[alpha], Cos[theta]*Cos[alpha], -Sin[alpha], -Sin[alpha]*d},
  {Sin[theta]*Sin[alpha], Cos[theta]*Sin[alpha], Cos[alpha], Cos[alpha]*d},
  {0, 0, 0, 1}
}

(* Compute the transformation matrices *)
transformationMatrices = Table[
  dhTransform @@ dhParameters[[i]],
  {i, Length[dhParameters]}
];

(* Function to compute overall forward kinematics up to a certain link *)
forwardKinematics[matrices_, link_] := Module[{fk},
  fk = IdentityMatrix[4];
  Do[
    fk = fk . matrices[[i]],
    {i, link}
  ];
  fk
];

(* Function to compute velocities by differentiating xyz components of forward kinematics *)
velocity[fk_] := Module[{vx, vy, vz},
  vx = D[fk[[1, 4]], t];
  vy = D[fk[[2, 4]], t];
  vz = D[fk[[3, 4]], t];
  {vx, vy, vz}
];

(* Function to compute angular velocities of each link *)
angularVelocity[transformationMatrices_] := Module[{angularVelocities, rotation, prevOmega},
  angularVelocities = Table[0, {Length[transformationMatrices]}];
  prevOmega = {0, 0, 0};
  
  Do[
    rotation = Transpose[transformationMatrices[[i, 1 ;; 3, 1 ;; 3]]];
    angularVelocities[[i]] = rotation . prevOmega + D[dhParameters[[i, 4]], t] * {0, 0, 1};
    prevOmega = angularVelocities[[i]],
    {i, Length[transformationMatrices]}
  ];
  angularVelocities
];

(* Function to compute final link angular velocity with respect to the base frame *)
finalLinkAngularVelocity[angularVelocities_, fkResult_, link_] := Module[{rotation},
  rotation = fkResult[[1 ;; 3, 1 ;; 3]];
  Simplify[rotation . angularVelocities[[link]]]
];

(* Display the transformation matrices with descriptions *)
Do[
  Print[StringJoin[ToString[i-1], "_", ToString[i], "T:"]];
  Print[MatrixForm[transformationMatrices[[i]]]],
  {i, Length[dhParameters]}
];

(* Calculate overall forward kinematics up to a certain link *)
fkResult = forwardKinematics[transformationMatrices, endLinkNumber];

(* Display the forward kinematics result *)
Print["____"];
Print[StringJoin[ToString[0], "_", ToString[endLinkNumber], "T:"]];
Print[MatrixForm[fkResult]];

(* Calculate velocities *)
velocities = Simplify[velocity[fkResult]];

(* Display the linear velocity of the end link *)
Print["____"];
Print[StringJoin[ToString[0], "_", ToString[endLinkNumber], "v:"]];
Print[MatrixForm[velocities]];

(* Calculate angular velocities *)
angularVelocities = Simplify[angularVelocity[transformationMatrices]];

(* Output angular velocities of each link *)
Print["____"];
Do[
  Print[StringJoin[ToString[i], "_", ToString[i], "w:"]];
  Print[MatrixForm[angularVelocities[[i]]]],
  {i, Length[angularVelocities]}
];

(* Calculate final link angular velocity with respect to the base frame *)
finalAngularVelocity = finalLinkAngularVelocity[angularVelocities, fkResult, endLinkNumber];

(* Output final link angular velocity with respect to the base frame *)
Print["____"];
Print[StringJoin[ToString[0], "_", ToString[endLinkNumber], "w:"]];
Print[MatrixForm[finalAngularVelocity]];




