PROGRAM MAIN

  USE GLOBAL, ONLY : WP, W0

  IMPLICIT NONE

!----------------------------------------------------------------------

  REAL(WP), PARAMETER :: A = 10., K = 1.

  LOGICAL        :: IS_S2_REFERENCE, LOCAL_NORM
  
  CHARACTER (1)  :: CHAR

  CHARACTER (20) :: S1_NAME, S2_NAME
  
  INTEGER        :: I, J, L, MT, NF_TF, NC
  REAL(WP)       :: DT, FMIN, FMAX, FF, TMP
  
  REAL(WP), DIMENSION (:,:)  , ALLOCATABLE :: S1, S2

  REAL(WP), DIMENSION (:)    , ALLOCATABLE ::   EM,   PM
  REAL(WP), DIMENSION (:,:  ), ALLOCATABLE ::  TEM,  TPM
  REAL(WP), DIMENSION (:,:  ), ALLOCATABLE ::  FEM,  FPM
  REAL(WP), DIMENSION (:,:,:), ALLOCATABLE :: TFEM, TFPM, CWT1, CWT2
  
!----------------------------------------------------------------------

  INTERFACE 

    SUBROUTINE TF_MISFITS_LOC (S1, S2, NC, DT, MT, FMIN, FMAX, NF_TF,  &
                                      TFEM,  TFPM,                     &
                                       TEM,   TPM,                     &
                                       FEM,   FPM,                     &
                                        EM,    PM,  CWT1, CWT2,        &
                                        IS_S2_REFERENCE )

      USE GLOBAL, ONLY: WP
      IMPLICIT NONE
      LOGICAL , INTENT(IN) :: IS_S2_REFERENCE
      INTEGER , INTENT(IN) :: MT, NF_TF, NC
      REAL(WP), INTENT(IN) :: DT, FMIN, FMAX
      REAL(WP), DIMENSION (NC,1:MT)   , INTENT(IN)  ::  S1, S2
      REAL(WP), DIMENSION (NC)        , INTENT(OUT) ::  EM,  PM
      REAL(WP), DIMENSION (NC,1:MT   ), INTENT(OUT) :: TEM, TPM
      REAL(WP), DIMENSION (NC,1:NF_TF), INTENT(OUT) :: FEM, FPM
      REAL(WP), DIMENSION (NC,1:MT,1:NF_TF), INTENT(OUT) :: TFEM, TFPM,&
                                                            CWT1, CWT2
    END SUBROUTINE TF_MISFITS_LOC

    SUBROUTINE TF_MISFITS_GLOB(S1, S2, NC, DT, MT, FMIN, FMAX, NF_TF,  &
                                      TFEM,  TFPM,                     &
                                       TEM,   TPM,                     &
                                       FEM,   FPM,                     &
                                        EM,    PM,  CWT1, CWT2,        &
                                        IS_S2_REFERENCE )

      USE GLOBAL, ONLY: WP
      IMPLICIT NONE
      LOGICAL , INTENT(IN) :: IS_S2_REFERENCE
      INTEGER , INTENT(IN) :: MT, NF_TF, NC
      REAL(WP), INTENT(IN) :: DT, FMIN, FMAX
      REAL(WP), DIMENSION (NC,1:MT)   , INTENT(IN)  ::  S1, S2
      REAL(WP), DIMENSION (NC)        , INTENT(OUT) ::  EM,  PM
      REAL(WP), DIMENSION (NC,1:MT   ), INTENT(OUT) :: TEM, TPM
      REAL(WP), DIMENSION (NC,1:NF_TF), INTENT(OUT) :: FEM, FPM
      REAL(WP), DIMENSION (NC,1:MT,1:NF_TF), INTENT(OUT) :: TFEM, TFPM,&
                                                            CWT1, CWT2
    END SUBROUTINE TF_MISFITS_GLOB

  END INTERFACE

!----------------------------------------------------------------------

  NAMELIST /INPUT/   MT, DT, FMIN, FMAX, S1_NAME, S2_NAME, NC,         &
                     IS_S2_REFERENCE, LOCAL_NORM

!----------------------------------------------------------------------

  FMIN = -999999.
  FMAX = -999999.
  IS_S2_REFERENCE = .FALSE.
  LOCAL_NORM      = .FALSE.
  NC   = 1
  
  OPEN ( 10, FILE='HF_TF-MISFIT_GOF', STATUS = 'OLD' )
  READ ( 10, NML = INPUT )

!=== ASSIGNMENT OF DEFAULT VALUES FOR FMIN AND FMAX  ( IF NOT PROVIDED )

  IF ( FMIN == -999999. ) THEN
    FMIN = 1./REAL(MT)/DT
  END IF
  IF ( FMAX == -999999. ) THEN
    FMAX = 1./2./DT
  END IF

!===================== COMPUTATION OF THE NUMBER OF SAMPLES IN FREQUENCY

  FF    = 1.+1./SQRT(2.)/W0/30.
  NF_TF = 1 + INT( LOG(FMAX/FMIN) / LOG(FF) )

!=============================================== ALLOCATION OF VARIABLES 
    
  ALLOCATE ( S1(NC,MT), S2(NC,MT), EM(NC), PM(NC) )
  
  ALLOCATE ( TEM (NC,1:MT   )        , TPM (NC,1:MT   ),               &
             FEM (NC,1:NF_TF)        , FPM (NC,1:NF_TF),               &
             TFEM(NC,1:MT   ,1:NF_TF), TFPM(NC,1:MT   ,1:NF_TF),       &
             CWT1(NC,1:MT   ,1:NF_TF), CWT2(NC,1:MT   ,1:NF_TF) )

!============================== READ INPUT FILES WITH SIGNALS S AND SREF

  OPEN (21, FILE= S1_NAME , STATUS='OLD')
  OPEN (22, FILE= S2_NAME , STATUS='OLD')
  OPEN (23, FILE='S1.DAT' , STATUS='NEW')
  OPEN (24, FILE='S2.DAT' , STATUS='NEW')

  DO I = 1, MT
    READ (21, *) TMP, ( S1 (J,I), J = 1, NC )
    READ (22, *) TMP, ( S2 (J,I), J = 1, NC )
    WRITE(23, *) DT*(I-1), ( S1 (J,I), J = 1, NC )
    WRITE(24, *) DT*(I-1), ( S2 (J,I), J = 1, NC )
  END DO

  CLOSE (21)
  CLOSE (22)
  CLOSE (23)
  CLOSE (24)

!======================= CALL SUBROUTINE FOR THE COMPUTATIONS OF MISFITS

  IF ( LOCAL_NORM ) THEN
    CALL TF_MISFITS_LOC (S1, S2, NC, DT, MT, FMIN, FMAX, NF_TF,        &
                                TFEM,  TFPM,                           &
                                 TEM,   TPM,                           &
                                 FEM,   FPM,                           &
                                  EM,    PM,   CWT1, CWT2,             &
                                  IS_S2_REFERENCE  )
  ELSE
    CALL TF_MISFITS_GLOB(S1, S2, NC, DT, MT, FMIN, FMAX, NF_TF,        &
                                TFEM,  TFPM,                           &
                                 TEM,   TPM,                           &
                                 FEM,   FPM,                           &
                                  EM,    PM,   CWT1, CWT2,             &
                                  IS_S2_REFERENCE  )
  END IF

!========================================================= WRITE RESULTS

  OPEN  ( 21, FILE='MISFIT-GOF.DAT', STATUS='NEW' )
  WRITE ( 21, *) FMIN, FMAX
  WRITE ( 21, *) NF_TF, MT
  WRITE ( 21, *) DT, NC
  WRITE ( 21, *) MAX(MAXVAL(ABS(S1)),MAXVAL(ABS(S2)))
  DO J = 1, NC
    WRITE ( 21, *) EM(J), PM(J)
  END DO
  DO J = 1, NC
    WRITE ( 21, *) A*EXP(-ABS(EM(J))**K), A*(1.-ABS(PM(J))**K)
  END DO
  WRITE ( 21, *) MAXVAL(ABS(TFEM)), MAXVAL(ABS(TFPM))
  WRITE ( 21, *) MAXVAL(ABS(FEM )), MAXVAL(ABS(FPM ))
  WRITE ( 21, *) MAXVAL(ABS(TEM )), MAXVAL(ABS(TPM ))
  WRITE ( 21, *) MAXVAL(ABS(CWT1)), MAXVAL(ABS(CWT2))
  CLOSE ( 21 )

  DO J = 1, NC
    WRITE ( CHAR, '(I1)' ) J
    OPEN  ( 21, FILE='TFEM'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( TFEM(J,I,L), I = 1, MT )
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TFPM'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( TFPM(J,I,L), I = 1, MT )
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TEM'//CHAR//'.DAT', STATUS='NEW' )
    DO I = 1, MT
      WRITE ( 21, * ) TEM(J,I)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TPM'//CHAR//'.DAT', STATUS='NEW' )
    DO I = 1, MT
      WRITE ( 21, * ) TPM(J,I)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='FEM'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) FEM(J,L)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='FPM'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) FPM(J,L)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TFRS1_'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( CWT1(J,I,L), I = 1, MT )
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TFRS2_'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( CWT2(J,I,L), I = 1, MT )
    END DO
    CLOSE ( 21 )
  END DO

!Computation of Goodness-of-fit criteria

  DO J = 1, NC
    WRITE ( CHAR, '(I1)' ) J
    OPEN  ( 21, FILE='TFEG'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( A*EXP(-ABS(TFEM(J,I,L))**K), I = 1, MT )
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TFPG'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) ( A*(1.-ABS(TFPM(J,I,L))**K), I = 1, MT )
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TEG'//CHAR//'.DAT', STATUS='NEW' )
    DO I = 1, MT
      WRITE ( 21, * ) A*EXP(-ABS(TEM(J,I))**K)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='TPG'//CHAR//'.DAT', STATUS='NEW' )
    DO I = 1, MT
      WRITE ( 21, * ) A*(1.-ABS(TPM(J,I))**K)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='FEG'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) A*EXP(-ABS(FEM(J,L))**K)
    END DO
    CLOSE ( 21 )

    OPEN ( 21, FILE='FPG'//CHAR//'.DAT', STATUS='NEW' )
    DO L = 1, NF_TF
      WRITE ( 21, * ) A*(1.-ABS(FPM(J,L))**K)
    END DO
    CLOSE ( 21 )
  END DO

END PROGRAM MAIN