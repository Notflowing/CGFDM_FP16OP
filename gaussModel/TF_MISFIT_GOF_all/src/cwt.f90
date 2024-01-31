FUNCTION CWT(F_V, MT, NF_TF, DFA, FF, FMIN)

  USE GLOBAL, ONLY: WP, PI, W0, KN, NF

  IMPLICIT NONE
!----------------------------------------------------------------------
  
  INTEGER , INTENT(IN)    :: MT, NF_TF
  REAL(WP), INTENT(IN)    :: DFA, FF, FMIN
  COMPLEX , INTENT(INOUT), DIMENSION(:) :: F_V
  
  INTEGER    :: I,J, NF21
  REAL(WP)   :: S, F
  COMPLEX, DIMENSION (1:MT, 1:NF_TF) :: CWT
  COMPLEX, DIMENSION (1:NF)          :: FPWV, FWV

!----------------------------------------------------------------------

  INTERFACE 

    FUNCTION     MORLET     ( S, FA )
      USE GLOBAL, ONLY: WP
      REAL(WP), INTENT(IN) :: S, FA
      COMPLEX(WP)          :: MORLET
    END FUNCTION MORLET  

  END INTERFACE

!----------------------------------------------------------------------

  NF21 = NF/2 + 1
  
  F = FMIN/FF

  DO I = 1, NF_TF

    F = F*FF
    S = W0 / (2.*PI*F)

    DO J = 1, NF21
      FPWV(J) = MORLET( S, REAL((J-1),WP)*DFA )
    END DO

    DO  J = NF21+1, NF
      FPWV(J) = (0._WP,0._WP)
    END DO

    DO J = 1, NF
      FWV (J) = F_V (J)*CONJG( FPWV(J) )
    END DO

    CALL FCOOLR (KN,FWV ,1._WP)

    CWT(1:MT,I) = FWV(1:MT)*DFA*SQRT(S)

  END DO
  
END FUNCTION CWT