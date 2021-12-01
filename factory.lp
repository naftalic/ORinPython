\ Model Factory
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  40 make[chair] + 50 make[table]
Subject To
 R[wood]: make[chair] + 2 make[table] <= 40
 R[labor]: 4 make[chair] + 3 make[table] <= 120
 R[packaging]: 0.2 make[chair] + 0.1 make[table] <= 5
Bounds
End
