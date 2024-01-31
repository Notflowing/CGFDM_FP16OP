SUBROUTINE FCOOLR(K,D,SN)
!
!    FAST FOURIER TRANSFORM
!
  IMPLICIT NONE

  INTEGER, PARAMETER :: SP = SELECTED_REAL_KIND (6,30)
  INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND (13,60)
  INTEGER, PARAMETER :: WP = SP

  INTEGER       :: LX, K, IL, I, NKK, LA, NCK, LCK, L2K, NW, ICK, LS,  &
                   J1, J, JH, JH1, ID, JJ

  REAL (KIND=WP), PARAMETER :: PI  = 3.141592654,                      &
                               PI2 = 2*PI

  REAL (KIND=WP):: SN, SH, Q1, Q2, FNW, AA, W1, W2

  INTEGER          , DIMENSION (32) :: INU
  REAL    (KIND=WP), DIMENSION ( 1) :: D
                       
  LX = 2**K
  Q1 = LX
  IL = LX
  SH = SN * PI2 / Q1
  DO I = 1, K
    IL     = IL / 2
    INU(I) = IL
  END DO
  NKK = 1
  DO LA = 1, K
    NCK = NKK
    NKK = NKK + NKK
    LCK = LX  / NCK
    L2K = LCK + LCK
    NW = 0
    DO ICK = 1, NCK
      FNW = NW
      AA  = SH * FNW
      W1  = COS(AA)
      W2  = SIN(AA)
      LS  = L2K * (ICK-1)
      DO I = 2, LCK, 2
        J1  = I  + LS
        J   = J1 - 1
        JH  = J  + LCK
        JH1 = JH + 1
        Q1  = D(JH)*W1 - D(JH1)*W2
        Q2  = D(JH)*W2 + D(JH1)*W1
        D(JH ) = D(J ) - Q1
        D(JH1) = D(J1) - Q2
        D(J  ) = D(J ) + Q1
        D(J1 ) = D(J1) + Q2
      END DO
      DO I = 2, K
        ID = INU(I)
        IL = ID + ID
        IF ( (NW-ID-IL*(NW/IL)) < 0 ) EXIT
        NW = NW - ID
      END DO
      NW = NW + ID
    END DO
  END DO
  NW = 0
  DO J = 1, LX
    IF ( (NW-J) >= 0 ) THEN 
      JJ  = NW  + NW + 1
      J1  = JJ  + 1
      JH1 = J   + J
      JH  = JH1 - 1
      Q1  = D(JJ)
      D(JJ ) = D(JH )
      D(JH ) = Q1
      Q1     = D(J1 )
      D(J1 ) = D(JH1)
      D(JH1) = Q1
    END IF
    DO I = 1, K
      ID = INU(I)
      IL = ID + ID
      IF ( (NW-ID-IL*(NW/IL)) < 0 ) EXIT
      NW = NW - ID
    END DO
    NW = NW + ID
  END DO
       
RETURN
END
