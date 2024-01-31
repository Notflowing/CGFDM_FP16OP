MODULE  GLOBAL

  IMPLICIT NONE

!----------------------------------------------------------------------

  INTEGER , PARAMETER :: SP = SELECTED_REAL_KIND (6,30)
  INTEGER , PARAMETER :: DP = SELECTED_REAL_KIND (13,60)
  INTEGER , PARAMETER :: WP = SP

  INTEGER , PARAMETER :: KN = 16
  INTEGER , PARAMETER :: NF = 2**KN
  
  REAL(WP), PARAMETER :: PI = 3.14159265359

  REAL(WP), PARAMETER :: W0 = 6.
  

!----------------------------------------------------------------------

END MODULE