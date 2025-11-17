# Test de balance
. /*=============================================================================
> 2. TEST DE BALANCE: TABLA 1 DEL PAPER (Sección 1.2)
> =============================================================================*/
. 
. display ""


. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "TABLA 1: TEST DE BALANCE - FORMALES VS INFORMALES"
TABLA 1: TEST DE BALANCE - FORMALES VS INFORMALES

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display ""


. 
. // Variables continuas (diferencias en medias)
. ttest ventas_k, by(ruc)

Two-sample t test with equal variances
------------------------------------------------------------------------------
   Group |     Obs        Mean    Std. err.   Std. dev.   [95% conf. interval]
---------+--------------------------------------------------------------------
Informal | 455,024    30.47287    .1073729    72.42887    30.26242    30.68332
Formal ( | 922,907    174.1698    .6497203    624.1736    172.8963    175.4432
---------+--------------------------------------------------------------------
Combined | 1377931    126.7178    .4403892    516.9522    125.8546    127.5809
---------+--------------------------------------------------------------------
    diff |           -143.6969    .9283787               -145.5165   -141.8773
------------------------------------------------------------------------------
    diff = mean(Informal) - mean(Formal ()                        t = -1.5e+02
H0: diff = 0                                     Degrees of freedom =  1.4e+06

    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0
 Pr(T < t) = 0.0000         Pr(|T| > |t|) = 0.0000          Pr(T > t) = 1.0000

. gen productividad_temp = productividad_x_trabajador / 1000
(45,472 missing values generated)

. ttest productividad_temp, by(ruc)

Two-sample t test with equal variances
------------------------------------------------------------------------------
   Group |     Obs        Mean    Std. err.   Std. dev.   [95% conf. interval]
---------+--------------------------------------------------------------------
Informal | 454,865    6.528034    .0145432     9.80844     6.49953    6.556539
Formal ( | 877,594    16.54478    .1510054    141.4618    16.24881    16.84074
---------+--------------------------------------------------------------------
Combined | 1332459    13.12534    .0996651    115.0456       12.93    13.32068
---------+--------------------------------------------------------------------
    diff |           -10.01674    .2100093               -10.42835   -9.605132
------------------------------------------------------------------------------
    diff = mean(Informal) - mean(Formal ()                        t = -47.6967
H0: diff = 0                                     Degrees of freedom =  1.3e+06

    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0
 Pr(T < t) = 0.0000         Pr(|T| > |t|) = 0.0000          Pr(T > t) = 1.0000

. drop productividad_temp

. ttest digital_score, by(ruc)

Two-sample t test with equal variances
------------------------------------------------------------------------------
   Group |     Obs        Mean    Std. err.   Std. dev.   [95% conf. interval]
---------+--------------------------------------------------------------------
Informal | 455,024    .1611629    .0005942    .4008089    .1599983    .1623275
Formal ( | 922,907    .4997394    .0007964    .7651268    .4981784    .5013004
---------+--------------------------------------------------------------------
Combined | 1377931    .3879338    .0005843    .6859329    .3867885    .3890791
---------+--------------------------------------------------------------------
    diff |           -.3385765    .0012086               -.3409452   -.3362077
------------------------------------------------------------------------------
    diff = mean(Informal) - mean(Formal ()                        t = -2.8e+02
H0: diff = 0                                     Degrees of freedom =  1.4e+06

    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0
 Pr(T < t) = 0.0000         Pr(|T| > |t|) = 0.0000          Pr(T > t) = 1.0000

. 
. // Variables categóricas (Chi²)
. tab region ruc, col nofreq chi2

    Región |
 geográfic |    Tenencia de RUC
     a del |     (Formalidad)
      Perú | Informal   Formal (c |     Total
-----------+----------------------+----------
     Costa |     54.58      62.17 |     59.67 
    Sierra |     36.93      30.27 |     32.47 
     Selva |      8.50       7.55 |      7.87 
-----------+----------------------+----------
     Total |    100.00     100.00 |    100.00 

          Pearson chi2(2) =  7.4e+03   Pr = 0.000

. tab sector ruc, col nofreq chi2

           |    Tenencia de RUC
    Sector |     (Formalidad)
 Económico | Informal   Formal (c |     Total
-----------+----------------------+----------
 Comercial |     75.54      59.00 |     64.49 
Productivo |      6.40       8.89 |      8.06 
 Servicios |     18.06      32.11 |     27.45 
-----------+----------------------+----------
     Total |    100.00     100.00 |    100.00 

          Pearson chi2(2) =  3.7e+04   Pr = 0.000

. 
. // Estadísticas por régimen (Sección 1.3)
. tabstat productividad_x_trabajador ventas_k, by(regimen) stat(mean n)

Summary statistics: Mean, N
Group variable: regimen (Régimen Tributario)

         regimen |  produc~r  ventas_k
-----------------+--------------------
         Ninguno |  6905.697  32.54221
                 |    550091    550603
-----------------+--------------------
Nuevo Régimen Ún |  10302.94  62.34235
                 |    484849    488391
-----------------+--------------------
Régimen Especial |  17681.78  130.1865
                 |     82780     83620
-----------------+--------------------
Régimen MYPE Tri |  26573.08  335.4519
                 |     90848     96882
-----------------+--------------------
Régimen General  |  38881.19  522.9757
                 |    123891    158435
-----------------+--------------------
           Total |  13125.34  126.7178
                 |   1332459   1377931
--------------------------------------

# Modelo principal
. /*=============================================================================
> 3. MODELO PRINCIPAL: TABLA 2 DEL PAPER (Sección 2.2)
> =============================================================================*/
. 
. display ""


. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "TABLA 2: MODELO LOGÍSTICO PRINCIPAL"
TABLA 2: MODELO LOGÍSTICO PRINCIPAL

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display ""


. 
. // Modelo completo con interacciones RUC×Región
. logit op2021_original c.ruc##(i.region) ventas_k i.sector i.sexo_gerente productividad_k digital_score tributos_k salarios_k i
> .tipo_local i.regimen, vce(cluster ciiu_4dig)

Iteration 0:  Log pseudolikelihood = -919800.61  
Iteration 1:  Log pseudolikelihood = -877875.63  
Iteration 2:  Log pseudolikelihood =  -877201.6  
Iteration 3:  Log pseudolikelihood =  -877190.2  
Iteration 4:  Log pseudolikelihood = -877190.14  
Iteration 5:  Log pseudolikelihood = -877190.14  

Logistic regression                                  Number of obs = 1,327,956
                                                     Wald chi2(19) =   1581.95
                                                     Prob > chi2   =    0.0000
Log pseudolikelihood = -877190.14                    Pseudo R2     =    0.0463

                                                       (Std. err. adjusted for 335 clusters in ciiu_4dig)
---------------------------------------------------------------------------------------------------------
                                        |               Robust
                        op2021_original | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
----------------------------------------+----------------------------------------------------------------
                                    ruc |  -.2160872   .0801533    -2.70   0.007    -.3731848   -.0589897
                                        |
                                 region |
                                Sierra  |  -.4231821   .0594118    -7.12   0.000     -.539627   -.3067371
                                 Selva  |  -.1344804   .0396178    -3.39   0.001    -.2121298   -.0568309
                                        |
                           region#c.ruc |
                                Sierra  |   .2622429   .0743699     3.53   0.000     .1164807    .4080052
                                 Selva  |   -.012587   .0628289    -0.20   0.841    -.1357294    .1105554
                                        |
                               ventas_k |   .0002815   .0000431     6.54   0.000      .000197    .0003659
                                        |
                                 sector |
                            Productivo  |  -.0426904    .103493    -0.41   0.680    -.2455329    .1601521
                             Servicios  |  -.0491771   .0809955    -0.61   0.544    -.2079254    .1095711
                                        |
                           sexo_gerente |
                                Hombre  |   .0321469   .0333163     0.96   0.335    -.0331519    .0974456
                        productividad_k |   .0131288   .0019953     6.58   0.000     .0092181    .0170395
                          digital_score |  -.1109475   .0235251    -4.72   0.000    -.1570558   -.0648393
                             tributos_k |  -.0001825   .0002759    -0.66   0.508    -.0007233    .0003583
                             salarios_k |   .0000544    .000067     0.81   0.417    -.0000769    .0001857
                                        |
                             tipo_local |
                             Alquilado  |  -.4589818   .0681475    -6.74   0.000    -.5925484   -.3254152
                                  Otro  |  -.4239409   .0724621    -5.85   0.000    -.5659641   -.2819178
                                        |
                                regimen |
Nuevo Régimen Único Simplificado (RUS)  |   .2830046   .0398454     7.10   0.000      .204909    .3611002
       Régimen Especial de Renta (RER)  |   .2302953   .0643892     3.58   0.000     .1040948    .3564957
         Régimen MYPE Tributario (RMT)  |   .6122861   .0785644     7.79   0.000     .4583027    .7662694
                  Régimen General (RG)  |   1.157923   .1793555     6.46   0.000      .806393    1.509454
                                        |
                                  _cons |   .1458162   .1126637     1.29   0.196    -.0750005    .3666329
---------------------------------------------------------------------------------------------------------

. 
. // Odds Ratios (IMPORTANTE: ejecutar ANTES de estimates store para que use el modelo activo)
. logit, or

Logistic regression                                  Number of obs = 1,327,956
                                                     Wald chi2(19) =   1581.95
                                                     Prob > chi2   =    0.0000
Log pseudolikelihood = -877190.14                    Pseudo R2     =    0.0463

                                                       (Std. err. adjusted for 335 clusters in ciiu_4dig)
---------------------------------------------------------------------------------------------------------
                                        |               Robust
                        op2021_original | Odds ratio   std. err.      z    P>|z|     [95% conf. interval]
----------------------------------------+----------------------------------------------------------------
                                    ruc |    .805665   .0645767    -2.70   0.007      .688538    .9427165
                                        |
                                 region |
                                Sierra  |   .6549594   .0389123    -7.12   0.000     .5829657     .735844
                                 Selva  |     .87417   .0346327    -3.39   0.001     .8088597    .9447538
                                        |
                           region#c.ruc |
                                Sierra  |   1.299842   .0966691     3.53   0.000     1.123536    1.503815
                                 Selva  |   .9874919   .0620431    -0.20   0.841     .8730788    1.116898
                                        |
                               ventas_k |   1.000281   .0000431     6.54   0.000     1.000197    1.000366
                                        |
                                 sector |
                            Productivo  |    .958208   .0991678    -0.41   0.680     .7822876    1.173689
                             Servicios  |   .9520125   .0771087    -0.61   0.544     .8122676    1.115799
                                        |
                           sexo_gerente |
                                Hombre  |   1.032669   .0344047     0.96   0.335     .9673916    1.102352
                        productividad_k |   1.013215   .0020217     6.58   0.000     1.009261    1.017185
                          digital_score |   .8949857   .0210546    -4.72   0.000     .8546564    .9372181
                             tributos_k |   .9998175   .0002759    -0.66   0.508      .999277    1.000358
                             salarios_k |   1.000054    .000067     0.81   0.417     .9999231    1.000186
                                        |
                             tipo_local |
                             Alquilado  |   .6319267   .0430642    -6.74   0.000     .5529164    .7222274
                                  Otro  |   .6544625   .0474237    -5.85   0.000     .5678125    .7543357
                                        |
                                regimen |
Nuevo Régimen Único Simplificado (RUS)  |   1.327111   .0528793     7.10   0.000     1.227413    1.434907
       Régimen Especial de Renta (RER)  |   1.258972   .0810642     3.58   0.000     1.109706    1.428315
         Régimen MYPE Tributario (RMT)  |   1.844644   .1449233     7.79   0.000     1.581388    2.151724
                  Régimen General (RG)  |   3.183316   .5709452     6.46   0.000     2.239814    4.524258
                                        |
                                  _cons |   1.156984     .13035     1.29   0.196     .9277431    1.442868
---------------------------------------------------------------------------------------------------------
Note: _cons estimates baseline odds.

. 
. // Guardar estimaciones para uso posterior
. estimates store modelo_principal

. 
. // Bondad de ajuste
. fitstat

Measures of Fit for logit of op2021_original

Log-Lik Intercept Only:  -919800.612     Log-Lik Full Model:      -877190.139
D(1327930):              1754380.278     LR(19):                    85220.946
                                         Prob > LR:                     0.000
McFadden's R2:                 0.046     McFadden's Adj R2:             0.046
Maximum Likelihood R2:         0.062     Cragg & Uhler's R2:            0.083
McKelvey and Zavoina's R2:     0.096     Efron's R2:                    0.062
Variance of y*:                3.639     Variance of error:             3.290
Count R2:                      0.591     Adj Count R2:                  0.155
AIC:                           1.321     AIC*n:                   1754432.278
BIC:                      -1.697e+07     BIC':                     -84953.062

. 
. // VIF test (multicolinealidad)
. quietly regress op2021_original productividad_k tributos_k salarios_k digital_score ventas_k i.sexo_gerente i.region i.sector 
> i.regimen i.tipo_local

. vif

    Variable |       VIF       1/VIF  
-------------+----------------------
productivi~k |      1.54    0.649982
  tributos_k |      1.18    0.844661
  salarios_k |      1.26    0.791966
digital_sc~e |      1.20    0.836262
    ventas_k |      1.57    0.636190
1.sexo_ger~e |      1.07    0.938462
      region |
          1  |      1.06    0.939990
          2  |      1.06    0.941813
      sector |
          1  |      1.07    0.933698
          2  |      1.15    0.869957
     regimen |
          1  |      1.24    0.808538
          2  |      1.15    0.868857
          3  |      1.24    0.805999
          4  |      1.37    0.731286
  tipo_local |
          1  |      1.09    0.916702
          2  |      1.03    0.974978
-------------+----------------------
    Mean VIF |      1.20

# Efectos marginales por ventas
. /*=============================================================================
> 5. EFECTOS MARGINALES (Secciones 2.3, 3.1, 3.2)
> =============================================================================*/
. 
. // Restaurar modelo principal
. estimates restore modelo_principal
(results modelo_principal are active now)

. 
. display ""


. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "EFECTOS MARGINALES POR REGIÓN (Sección 2.3)"
EFECTOS MARGINALES POR REGIÓN (Sección 2.3)

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display ""


. 
. // Efecto marginal de RUC por región
. margins region, dydx(ruc)

Average marginal effects                             Number of obs = 1,327,956
Model VCE: Robust

Expression: Pr(op2021_original), predict()
dy/dx wrt:  ruc

------------------------------------------------------------------------------
             |            Delta-method
             |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
ruc          |
      region |
      Costa  |  -.0509367   .0188255    -2.71   0.007    -.0878339   -.0140394
     Sierra  |   .0107841   .0088621     1.22   0.224    -.0065853    .0281535
      Selva  |  -.0540898   .0224143    -2.41   0.016    -.0980211   -.0101586
------------------------------------------------------------------------------

. marginsplot, title("Efecto marginal de RUC por región") ///
>             ytitle("Cambio en Pr(Supervivencia)") xtitle("Región")

Variables that uniquely identify margins: region

. 
. // Tests de heterogeneidad
. test 1.region#c.ruc 2.region#c.ruc  // H0: Efectos iguales en todas regiones

 ( 1)  [op2021_original]1.region#c.ruc = 0
 ( 2)  [op2021_original]2.region#c.ruc = 0

           chi2(  2) =   12.81
         Prob > chi2 =    0.0016

. test 1.region#c.ruc = 0             // H0: Sierra = Costa

 ( 1)  [op2021_original]1.region#c.ruc = 0

           chi2(  1) =   12.43
         Prob > chi2 =    0.0004

. test 2.region#c.ruc = 0             // H0: Selva = Costa

 ( 1)  [op2021_original]2.region#c.ruc = 0

           chi2(  1) =    0.04
         Prob > chi2 =    0.8412

. test 1.region#c.ruc = 2.region#c.ruc // H0: Sierra = Selva

 ( 1)  [op2021_original]1.region#c.ruc - [op2021_original]2.region#c.ruc = 0

           chi2(  1) =    9.01
         Prob > chi2 =    0.0027

. 
. display ""


. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "EFECTOS POR VENTAS (Sección 3.1)"
EFECTOS POR VENTAS (Sección 3.1)

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════
. // Efecto de RUC según escala de ventas
. margins region, dydx(ruc) at(ventas_k = (0 (1000) 7478))

Average marginal effects                             Number of obs = 1,327,956
Model VCE: Robust

Expression: Pr(op2021_original), predict()
dy/dx wrt:  ruc
1._at: ventas_k =    0
2._at: ventas_k = 1000
3._at: ventas_k = 2000
4._at: ventas_k = 3000
5._at: ventas_k = 4000
6._at: ventas_k = 5000
7._at: ventas_k = 6000
8._at: ventas_k = 7000

------------------------------------------------------------------------------
             |            Delta-method
             |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
ruc          |
  _at#region |
    1#Costa  |  -.0513015   .0189564    -2.71   0.007    -.0884553   -.0141476
   1#Sierra  |   .0108503   .0089157     1.22   0.224     -.006624    .0283247
    1#Selva  |  -.0544413   .0225518    -2.41   0.016    -.0986421   -.0102406
    2#Costa  |  -.0497081   .0183121    -2.71   0.007    -.0855993    -.013817
   2#Sierra  |   .0108579   .0089243     1.22   0.224    -.0066333    .0283491
    2#Selva  |  -.0536949   .0221139    -2.43   0.015    -.0970373   -.0103525
    3#Costa  |  -.0465261   .0172144    -2.70   0.007    -.0802658   -.0127865
   3#Sierra  |   .0104975   .0086521     1.21   0.225    -.0064604    .0274553
    3#Selva  |  -.0511375   .0210983    -2.42   0.015    -.0924894   -.0097855
    4#Costa  |  -.0421324   .0158309    -2.66   0.008    -.0731604   -.0111043
   4#Sierra  |   .0098095   .0081321     1.21   0.228    -.0061291    .0257481
    4#Selva  |  -.0470755    .019664    -2.39   0.017    -.0856162   -.0085348
    5#Costa  |  -.0370049   .0143236    -2.58   0.010    -.0650787   -.0089312
   5#Sierra  |   .0088732   .0074253     1.19   0.232    -.0056802    .0234265
    5#Selva  |  -.0419748   .0179884    -2.33   0.020    -.0772313   -.0067182
    6#Costa  |  -.0316222   .0128027    -2.47   0.014    -.0567149   -.0065294
   6#Sierra  |   .0077877   .0066055     1.18   0.238    -.0051589    .0207343
    6#Selva  |  -.0363541   .0162126    -2.24   0.025    -.0681302    -.004578
    7#Costa  |  -.0263835   .0113223    -2.33   0.020    -.0485748   -.0041922
   7#Sierra  |   .0066521   .0057433     1.16   0.247    -.0046046    .0179088
    7#Selva  |  -.0306866   .0144234    -2.13   0.033    -.0589558   -.0024173
    8#Costa  |  -.0215694    .009904    -2.18   0.029    -.0409809    -.002158
   8#Sierra  |   .0055489   .0048962     1.13   0.257    -.0040474    .0151452
    8#Selva  |  -.0253353   .0126687    -2.00   0.046    -.0501655   -.0005051
------------------------------------------------------------------------------

. marginsplot, title("Efecto de RUC según escala de ventas") ///
>             ytitle("Efecto marginal de RUC") xtitle("Ventas (en miles)")

Variables that uniquely identify margins: ventas_k region

# Analisis por regimen tributario
. /*=============================================================================
> 6. ANÁLISIS POR RÉGIMEN TRIBUTARIO (Sección 4.2)
> =============================================================================*/
. 
. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "PROBABILIDADES POR RÉGIMEN TRIBUTARIO (Sección 4.2)"
PROBABILIDADES POR RÉGIMEN TRIBUTARIO (Sección 4.2)

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. 
. // Probabilidades predichas por régimen
. margins regimen

Predictive margins                                   Number of obs = 1,327,956
Model VCE: Robust

Expression: Pr(op2021_original), predict()

---------------------------------------------------------------------------------------------------------
                                        |            Delta-method
                                        |     Margin   std. err.      z    P>|z|     [95% conf. interval]
----------------------------------------+----------------------------------------------------------------
                                regimen |
                               Ninguno  |    .455739   .0140756    32.38   0.000     .4281514    .4833266
Nuevo Régimen Único Simplificado (RUS)  |   .5237031   .0106588    49.13   0.000     .5028122     .544594
       Régimen Especial de Renta (RER)  |   .5110246   .0112676    45.35   0.000     .4889405    .5331087
         Régimen MYPE Tributario (RMT)  |   .6017298   .0147702    40.74   0.000     .5727807     .630679
                  Régimen General (RG)  |   .7195014   .0316938    22.70   0.000     .6573827      .78162
---------------------------------------------------------------------------------------------------------

. marginsplot, title("Probabilidad de supervivencia por régimen tributario") ///
>             ytitle("Pr(Supervivencia)") xlabel(, angle(45))

Variables that uniquely identify margins: regimen

. 
. // Test de diferencias entre regímenes
. test 1.regimen 2.regimen 3.regimen 4.regimen

 ( 1)  [op2021_original]1.regimen = 0
 ( 2)  [op2021_original]2.regimen = 0
 ( 3)  [op2021_original]3.regimen = 0
 ( 4)  [op2021_original]4.regimen = 0

           chi2(  4) =  142.04
         Prob > chi2 =    0.0000

. 
. // Probabilidades condicionales a formalización (RUC=1)
. margins regimen, at(ruc=1)

Predictive margins                                   Number of obs = 1,327,956
Model VCE: Robust

Expression: Pr(op2021_original), predict()
At: ruc = 1

---------------------------------------------------------------------------------------------------------
                                        |            Delta-method
                                        |     Margin   std. err.      z    P>|z|     [95% conf. interval]
----------------------------------------+----------------------------------------------------------------
                                regimen |
                               Ninguno  |   .4455927   .0138591    32.15   0.000     .4184294    .4727559
Nuevo Régimen Único Simplificado (RUS)  |   .5136868   .0109419    46.95   0.000     .4922412    .5351325
       Régimen Especial de Renta (RER)  |   .5009551   .0112297    44.61   0.000     .4789454    .5229648
         Régimen MYPE Tributario (RMT)  |   .5923208   .0138251    42.84   0.000     .5652242    .6194174
                  Régimen General (RG)  |   .7118589   .0313458    22.71   0.000     .6504223    .7732955
---------------------------------------------------------------------------------------------------------

. 
. // Comparaciones por pares
. margins regimen, at(ruc=1) pwcompare(effects)

Pairwise comparisons of predictive margins           Number of obs = 1,327,956
Model VCE: Robust

Expression: Pr(op2021_original), predict()
At: ruc = 1

--------------------------------------------------------------------------------------------------------------------------
                                                         |            Delta-method    Unadjusted           Unadjusted
                                                         |   Contrast   std. err.      z    P>|z|     [95% conf. interval]
---------------------------------------------------------+----------------------------------------------------------------
                                                 regimen |
      Nuevo Régimen Único Simplificado (RUS) vs Ninguno  |   .0680942   .0094611     7.20   0.000     .0495507    .0866376
             Régimen Especial de Renta (RER) vs Ninguno  |   .0553624   .0153731     3.60   0.000     .0252317    .0854932
               Régimen MYPE Tributario (RMT) vs Ninguno  |   .1467281   .0184026     7.97   0.000     .1106597    .1827964
                        Régimen General (RG) vs Ninguno  |   .2662662   .0368646     7.22   0.000      .194013    .3385195
                        Régimen Especial de Renta (RER)  |
                                                     vs  |
                 Nuevo Régimen Único Simplificado (RUS)  |  -.0127317    .009217    -1.38   0.167    -.0307968    .0053333
                          Régimen MYPE Tributario (RMT)  |
                                                     vs  |
                 Nuevo Régimen Único Simplificado (RUS)  |   .0786339   .0125885     6.25   0.000     .0539609     .103307
                                   Régimen General (RG)  |
                                                     vs  |
                 Nuevo Régimen Único Simplificado (RUS)  |   .1981721   .0340461     5.82   0.000     .1314429    .2649012
                          Régimen MYPE Tributario (RMT)  |
                                                     vs  |
                        Régimen Especial de Renta (RER)  |   .0913657   .0098692     9.26   0.000     .0720225    .1107089
Régimen General (RG) vs Régimen Especial de Renta (RER)  |   .2109038   .0313296     6.73   0.000     .1494989    .2723087
  Régimen General (RG) vs Régimen MYPE Tributario (RMT)  |   .1195382   .0311081     3.84   0.000     .0585674    .1805089
--------------------------------------------------------------------------------------------------------------------------

# Diagnostico del modelp
. /*=============================================================================
> 7. DIAGNÓSTICOS DEL MODELO (Sección 5.2)
> =============================================================================*/
. 
. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. display "DIAGNÓSTICOS: CAPACIDAD PREDICTIVA"
DIAGNÓSTICOS: CAPACIDAD PREDICTIVA

. display "══════════════════════════════════════════════════════════"
══════════════════════════════════════════════════════════

. 
. // Matriz de clasificación
. estat classification

Logistic model for op2021_original

              -------- True --------
Classified |         D            ~D  |      Total
-----------+--------------------------+-----------
     +     |    410288        268362  |     678650
     -     |    274754        374552  |     649306
-----------+--------------------------+-----------
   Total   |    685042        642914  |    1327956

Classified + if predicted Pr(D) >= .5
True D defined as op2021_original != 0
--------------------------------------------------
Sensitivity                     Pr( +| D)   59.89%
Specificity                     Pr( -|~D)   58.26%
Positive predictive value       Pr( D| +)   60.46%
Negative predictive value       Pr(~D| -)   57.68%
--------------------------------------------------
False + rate for true ~D        Pr( +|~D)   41.74%
False - rate for true D         Pr( -| D)   40.11%
False + rate for classified +   Pr(~D| +)   39.54%
False - rate for classified -   Pr( D| -)   42.32%
--------------------------------------------------
Correctly classified                        59.10%
--------------------------------------------------

. 
. // Curva ROC
. lroc, title("Curva ROC - Capacidad predictiva del modelo")

Logistic model for op2021_original

Number of observations =  1327956
Area under ROC curve   =   0.6367

. 
. // Criterios de información
. estat ic

Akaike's information criterion and Bayesian information criterion

-----------------------------------------------------------------------------
       Model |          N   ll(null)  ll(model)      df        AIC        BIC
-------------+---------------------------------------------------------------
modelo_pri~l |  1,327,956  -919800.6  -877190.1      20    1754420    1754662
-----------------------------------------------------------------------------
Note: BIC uses N = number of observations. See [R] IC note.