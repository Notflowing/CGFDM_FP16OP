SUBROUTINE TF_MISFITS_LOC (S1, S2, NC, DT, MT, FMIN, FMAX, NF_TF,      &
                                  TFEM,  TFPM,                         &
                                   TEM,   TPM,                         &
                                   FEM,   FPM,                         &
                                    EM,    PM,  CWT1, CWT2,            &
                                    IS_S2_REFERENCE  )

  USE GLOBAL, ONLY: WP, PI, W0, KN, NF

  IMPLICIT NONE

!----------------------------------------------------------------------
  LOGICAL , INTENT(IN) :: IS_S2_REFERENCE
  INTEGER , INTENT(IN) :: MT, NF_TF, NC
  REAL(WP), INTENT(IN) :: DT, FMIN, FMAX
  
  REAL(WP), DIMENSION (NC,1:MT)   , INTENT(IN)  :: S1, S2
  
  REAL(WP), DIMENSION (NC)        , INTENT(OUT) ::  EM, PM
  REAL(WP), DIMENSION (NC,1:MT   ), INTENT(OUT) :: TEM, TPM
  REAL(WP), DIMENSION (NC,1:NF_TF), INTENT(OUT) :: FEM, FPM

  REAL(WP), DIMENSION (NC,1:MT,1:NF_TF), INTENT(OUT) ::  TFEM, TFPM,   &
                                                         CWT1, CWT2

  INTEGER            :: I, J, L

  REAL(WP)           :: NOMSE, NOMSP, DENOMS, DF, FF, MAXDENOM, MAXTF, MV
  
  REAL(WP), DIMENSION(NC,1:MT,1:NF_TF)    :: DE , DP

  REAL(WP), DIMENSION(NC,1:MAX(MT,NF_TF)) :: DET, DPT, DEF, DPF
                                          
  COMPLEX , DIMENSION(NC,NF)           :: SS1, SS2
  COMPLEX , DIMENSION(NC,1:MT,1:NF_TF) :: WV1, WV2, WV_REF

!----------------------------------------------------------------------

  INTERFACE 

    FUNCTION CWT(F_V, MT, NF_TF, DFA, FF, FMIN)
      USE GLOBAL, ONLY: WP
      IMPLICIT NONE
      INTEGER , INTENT(IN)    :: MT, NF_TF
      REAL(WP), INTENT(IN)    :: DFA, FF, FMIN
      COMPLEX , INTENT(INOUT), DIMENSION(:) :: F_V
      COMPLEX, DIMENSION (1:MT, 1:NF_TF) :: CWT
    END FUNCTION CWT
    
  END INTERFACE

!----------------------------------------------------------------------

! COMPUTATION OF CWT

  DF  = 1./(DT*REAL(NF))

  FF = EXP( LOG(FMAX/FMIN) / REAL(NF_TF - 1) )
   
  DO J = 1, NC
    SS1 (J,:) = CMPLX(0.,0.)
    SS2 (J,:) = CMPLX(0.,0.)
    DO I = 1, MT
      SS1 (J,I) = CMPLX( S1 (J,I), 0. )
      SS2 (J,I) = CMPLX( S2 (J,I), 0. )
    END DO
    CALL FCOOLR (KN, SS1 (J,:), -1.)
    CALL FCOOLR (KN, SS2 (J,:), -1.)
    
    SS1 = DT*SS1
    SS2 = DT*SS2
    
    WV1 (J,1:MT,1:NF_TF) = CWT(SS1 (J,:), MT ,NF_TF, DF, FF, FMIN)
    WV2 (J,1:MT,1:NF_TF) = CWT(SS2 (J,:), MT, NF_TF, DF, FF, FMIN)

    IF ( .NOT.(IS_S2_REFERENCE) .AND.                                  &
         ( MAXVAL (ABS(WV1(J,:,:))) < MAXVAL (ABS(WV2(J,:,:))) ) ) THEN
      WV_REF (J,:,:) = WV1 (J,:,:)
    ELSE
      WV_REF (J,:,:) = WV2 (J,:,:)
    END IF

  END DO

  CWT1 = ABS(WV1)*ABS(WV1)
  CWT2 = ABS(WV2)*ABS(WV2)
  
! COMPUTATION OF TFEM AND TFPM

  DE = ABS(WV1) - ABS(WV2)
  
  DO J = 1, NC
    DO I = 1, MT
      DO L = 1, NF_TF
        IF ( ( ABS(WV1(J,I,L)) == 0. ) .OR. ( ABS(WV2(J,I,L)) == 0. ) ) THEN
          DP(J,I,L) = 0.
        ELSE
          DP(J,I,L) =  ATAN2( IMAG(WV1 (J,I,L)/WV2 (J,I,L) ),          &
                              REAL(WV1 (J,I,L)/WV2 (J,I,L) ) ) / PI
        END IF
        DP(J,I,L) = ABS(WV_REF(J,I,L)) * DP(J,I,L)
      END DO
    END DO
  END DO
  
    
  MV = MAXVAL(ABS(WV_REF))
  DO J = 1, NC
    DO I = 1, MT
      DO L = 1, NF_TF
        IF ( ABS(WV_REF(J,I,L)) < 0.000*MV ) THEN
          TFEM(J,I,L) = -2.
        ELSE
          TFEM(J,I,L) = DE(J,I,L) / ABS(WV_REF(J,I,L))
        ENDIF
      END DO
    END DO
  END DO
  TFPM = DP / ABS(WV_REF)
        
! COMPUTATION OF AUXILIARY VARIABLES

  DO I = 1, MT
    DET  (:,I) = 0.
    DPT  (:,I) = 0.
    DO L = 1, NF_TF
      DET (:,I) = DET (:,I) + DE(:,I,L)
      DPT (:,I) = DPT (:,I) + DP(:,I,L)
    END DO
    DET (:,I) = DET (:,I)/REAL(NF_TF)
    DPT (:,I) = DPT (:,I)/REAL(NF_TF)
  END DO

  DO L = 1, NF_TF
    DEF   (:,L) = 0.
    DPF   (:,L) = 0.
    DO I = 1, MT
      DEF (:,L) = DEF (:,L) + DE(:,I,L)
      DPF (:,L) = DPF (:,L) + DP(:,I,L)
    END DO
    DEF (:,L) = DEF (:,L)/REAL(MT)
    DPF (:,L) = DPF (:,L)/REAL(MT)
  END DO  
  
! COMPUTATION OF TEM ANF TPM

  DO I = 1, MT
    DO J = 1, NC
      DENOMS = 0.
      DO L = 1, NF_TF
        DENOMS = DENOMS + ABS(WV_REF(J,I,L))
      END DO
      DENOMS = DENOMS / REAL(NF_TF)

      TEM(J,I) = DET (J,I) / DENOMS
      TPM(J,I) = DPT (J,I) / DENOMS
    END DO
  END DO

! COMPUTATION OF FEM AND FPM

  DO L = 1, NF_TF
    DO J = 1, NC
      DENOMS = 0.
      DO I = 1, MT
        DENOMS = DENOMS + ABS(WV_REF(J,I,L))
      END DO
      DENOMS = DENOMS / REAL(MT)

      FEM(J,L) = DEF (J,L) / DENOMS
      FPM(J,L) = DPF (J,L) / DENOMS
    END DO
  END DO

! COMPUTATION OF EM AND PM
 
  DO J = 1, NC
    NOMSE  = 0.
    NOMSP  = 0.
    DENOMS = 0.

    DO I = 1, MT
      DO L = 1, NF_TF

        NOMSE  = NOMSE  + ABS(DE    (J,I,L))*ABS(DE    (J,I,L))
        NOMSP  = NOMSP  + ABS(DP    (J,I,L))*ABS(DP    (J,I,L))
        DENOMS = DENOMS + ABS(WV_REF(J,I,L))*ABS(WV_REF(J,I,L))
                          
      END DO
    END DO
  
    EM(J) = SQRT ( NOMSE / DENOMS )
    PM(J) = SQRT ( NOMSP / DENOMS )

  END DO

END SUBROUTINE  TF_MISFITS_LOC