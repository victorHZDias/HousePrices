import pandas as pd
import polars as pl
import lux
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#Análise exploratória dos dados e encoding

baseTest=pd.read_csv('test.csv',sep=",")

len(baseTest.columns)

baseTest.info()

baseTest.head()

baseTest.describe()

def oneHotEncoding(coluna):
    ohe = OneHotEncoder()
    ohe=ohe.fit(coluna)
    dadosArray=ohe.transform(coluna).toarray()
    colunas=ohe.get_feature_names_out()
    dfOhe=pd.DataFrame(dadosArray,columns=colunas)
    return dfOhe

def ordinalEncoding(coluna):
    ode = OrdinalEncoder()
    ode=ode.fit(coluna)
    dadosArray=ode.transform(coluna)
    colunas=ode.get_feature_names_out()
    dfOde=pd.DataFrame(dadosArray,columns=colunas)
    return dfOde
#__________________________________________________________________________________________________
# MSSubClass: Identifies the type of dwelling involved in the sale.	

categories={20:	"1-STORY 1946 & NEWER ALL STYLES",
        30:	"1-STORY 1945 & OLDER",
        40:	"1-STORY W/FINISHED ATTIC ALL AGES",
        45:	"1-1/2 STORY - UNFINISHED ALL AGES",
        50:	"1-1/2 STORY FINISHED ALL AGES",
        60:	"2-STORY 1946 & NEWER",
        70:	"2-STORY 1945 & OLDER",
        75:	"2-1/2 STORY ALL AGES",
        80:	"SPLIT OR MULTI-LEVEL",
        85:	"SPLIT FOYER",
        90:	"DUPLEX - ALL STYLES AND AGES",
       120:	"1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
       150:	"1-1/2 STORY PUD - ALL AGES",
       160:	"2-STORY PUD - 1946 & NEWER",
       180:	"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
       190:	"2 FAMILY CONVERSION - ALL STYLES AND AGES"
}

baseTest['MSSubClass']=baseTest['MSSubClass'].replace(categories)

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['MSSubClass']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('MSSubClass',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna MSZoning:
#MSZoning: Identifies the general zoning classification of the sale.
    #    A	Agriculture
    #    C	Commercial
    #    FV	Floating Village Residential
    #    I	Industrial
    #    RH	Residential High Density
    #    RL	Residential Low Density
    #    RP	Residential Low Density Park 
    #    RM	Residential Medium Density

baseTest['MSZoning'].value_counts()
baseTest[baseTest['MSZoning'].isnull()]
baseTest['MSZoning']=baseTest['MSZoning'].fillna('RL')


#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['MSZoning']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('MSZoning',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotFrontage:
#Valores Nulos substituidos pela mediana
baseTest['LotFrontage']=baseTest['LotFrontage'].fillna(baseTest['LotFrontage'].median())

#__________________________________________________________________________________________________

# Tratamento coluna Street:
#Colua categórica com dua opções, encoding para 0 e 1
baseTest['Street'].value_counts()

baseTest.query("Street == 'Grvl'")

baseTest.query("Street == 'Pave'")

baseTest['Street']=baseTest['Street'].apply(lambda x: 0 if x=='Pave' else 1)

#__________________________________________________________________________________________________

# Tratamento coluna Alley:
#Maioria dos valores nulo, não será considerada no modelo
baseTest['Alley'].value_counts()
baseTest=baseTest.drop('Alley',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotShape:
# LotShape: General shape of property

#        Reg	Regular	
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['LotShape']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('LotShape',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LandContour:
# LandContour: Flatness of the property

#        Lvl	Near Flat/Level	
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['LandContour']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('LandContour',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Utilities:
#Maioria dos valores é AllPub, não será considerada no modelo
baseTest=baseTest.drop('Utilities',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotConfig:
# LotConfig: Lot configuration

#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['LotConfig']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('LotConfig',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LandSlope:
# LandSlope: Slope of property
		
#        Gtl	Gentle slope
#        Mod	Moderate Slope	
#        Sev	Severe Slope
       
#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['LandSlope']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('LandSlope',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Neighborhood:
# Neighborhood: Physical locations within Ames city limits

#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	Northwest Ames
#        OldTown	Old Town
#        SWISU	South & West of Iowa State University
#        Sawyer	Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	Timberland
#        Veenker	Veenker

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['Neighborhood']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Neighborhood',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Condition1:
# Condition1: Proximity to various conditions
	
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['Condition1']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Condition1',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Condition2:
#Similar a coluna condition1, não será considerada no modelo
baseTest=baseTest.drop('Condition2',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BldgType:
# BldgType: Type of dwelling
		
#        1Fam	Single-family Detached	
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['BldgType']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('BldgType',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna HouseStyle:
#Similar a coluna MSSubClass, não será considerada no modelo
baseTest=baseTest.drop('HouseStyle',axis=1)


#__________________________________________________________________________________________________

# Tratamento colunas YearBuilt e YearRemodAdd

baseTest['YearBuilt'].sort_values().unique()
baseTest['YearRemodAdd'].sort_values().unique()

#Função para transformar os dados de ano em categorias por períodos:
def clasAnos(x):
        
    if x >=1800 and x <=1899: 
        return '1800 a 1899'
    elif x >=1900 and x <=1999:
        return '1900 a 1900'
    else:
        return 'acima de 2000'
        
baseTest['YearBuilt']=baseTest['YearBuilt'].apply(clasAnos)
baseTest['YearRemodAdd']=baseTest['YearRemodAdd'].apply(clasAnos)

#Fazendo o OneHot Encoding dos dados categoricos criados
dfOhe=oneHotEncoding(baseTest[['YearBuilt']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('YearBuilt',axis=1)
dfOhe=oneHotEncoding(baseTest[['YearRemodAdd']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('YearRemodAdd',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna RoofStyle
# RoofStyle: Type of roof

#        Flat	Flat
#        Gable	Gable
#        Gambrel	Gabrel (Barn)
#        Hip	Hip
#        Mansard	Mansard
#        Shed	Shed

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['RoofStyle']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('RoofStyle',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna RoofMatl
baseTest['RoofMatl'].value_counts()
#Similar a coluna RoofStyle, não será considerada no modelo
baseTest=baseTest.drop('RoofMatl',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Exterior1st
# Exterior1st: Exterior covering on house

#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast	
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles

#Coluna Categórica, OneHot Encoding dos dados
baseTest[baseTest['Exterior1st'].isna()]
baseTest['Exterior1st']=baseTest['Exterior1st'].fillna('VinylSd')

baseTest['Exterior1st'].value_counts()
baseTest.iloc[691,:10]

dfOhe=oneHotEncoding(baseTest[['Exterior1st']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Exterior1st',axis=1)
baseTest['Exterior1st_ImStucc']=0
baseTest['Exterior1st_Stone']=0
#__________________________________________________________________________________________________

# Tratamento coluna Exterior2nd
#Similar a coluna Exterior1st, não será considerada no modelo
baseTest=baseTest.drop('Exterior2nd',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna MasVnrType
baseTest['MasVnrType'].value_counts()
baseTest['MasVnrType']=baseTest['MasVnrType'].fillna('não tem')
baseTest.loc[baseTest['MasVnrType'].isnull()]

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['MasVnrType']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('MasVnrType',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna MasVnrArea
baseTest['MasVnrArea'].value_counts()
baseTest.loc[baseTest['MasVnrArea'].isnull()]
baseTest['MasVnrArea']=baseTest['MasVnrArea'].fillna(baseTest['MasVnrArea'].median())

#__________________________________________________________________________________________________
 
# Tratamento coluna ExterQual
# ExterQual: Evaluates the quality of the material on the exterior 
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['ExterQual']])
dfOde=dfOde.rename({'ExterQual':'ExterQual_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest=baseTest.drop('ExterQual',axis=1)

# Tratamento coluna ExterCond
baseTest['ExterCond'].value_counts()
# ExterCond: Evaluates the present condition of the material on the exterior
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['ExterCond']])
dfOde=dfOde.rename({'ExterCond':'ExterCond_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest=baseTest.drop('ExterCond',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Foundation
baseTest['Foundation'].value_counts()
# Foundation: Type of foundation
		
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete	
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['Foundation']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Foundation',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtQual
baseTest['BsmtQual'].value_counts()
baseTest['BsmtQual']=baseTest['BsmtQual'].fillna("No Basement")
# BsmtQual: Evaluates the height of the basement

#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['BsmtQual']])
dfOde=dfOde.rename({'BsmtQual':'BsmtQual_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest=baseTest.drop('BsmtQual',axis=1)

baseTest['BsmtQual_Ode'].value_counts()

#__________________________________________________________________________________________________

# Tratamento coluna BsmtCond
baseTest['BsmtCond'].value_counts()
baseTest['BsmtCond']=baseTest['BsmtCond'].fillna("No Basement")
# BsmtCond: Evaluates the general condition of the basement

#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['BsmtCond']])
dfOde=dfOde.rename({'BsmtCond':'BsmtCond_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest=baseTest.drop('BsmtCond',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtExposure
baseTest['BsmtExposure'].value_counts()
baseTest['BsmtExposure']=baseTest['BsmtExposure'].fillna("No Basement")
baseTest.loc[baseTest['BsmtExposure']=='No','BsmtExposure']='No Exposure'

# BsmtExposure: Refers to walkout or garden level walls

#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement

dfOde=ordinalEncoding(baseTest[['BsmtExposure']])
dfOde=dfOde.rename({'BsmtExposure':'BsmtExposure_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtExposure',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtFinType1: Rating of basement finished area
baseTest['BsmtFinType1'].value_counts()
baseTest[baseTest['BsmtFinType1'].isnull()]
baseTest['BsmtFinType1']=baseTest['BsmtFinType1'].fillna("No Basement")

    #    GLQ	Good Living Quarters
    #    ALQ	Average Living Quarters
    #    BLQ	Below Average Living Quarters	
    #    Rec	Average Rec Room
    #    LwQ	Low Quality
    #    Unf	Unfinshed
    #    NA	No Basement
    
#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['BsmtFinType1']])
dfOde=dfOde.rename({'BsmtFinType1':'BsmtFinType1_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['BsmtFinType1_Ode'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtFinType1',axis=1)
baseTest=baseTest.drop('BsmtFinType1_Ode',axis=1)

#__________________________________________________________________________________________________
# Tratamento coluna BsmtFinSF1: Type 1 finished square feet
#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtFinSF1',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtFinType2: Rating of basement finished area (if multiple types)

#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtFinType2',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtFinSF2: Type 2 finished square feet

baseTest['BsmtFinSF2'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtFinSF2',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna BsmtUnfSF: Unfinished square feet of basement area

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtUnfSF',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna TotalBsmtSF: Total square feet of basement area

baseTest['TotalBsmtSF'].value_counts()
baseTest['TotalBsmtSF']=baseTest['TotalBsmtSF'].fillna(baseTest['TotalBsmtSF'].median())

# __________________________________________________________________________________________________

# Tratamento coluna Heating: Type of heating
		
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace	
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace

baseTest['Heating'].value_counts()
dfOhe=oneHotEncoding(baseTest[['Heating']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Heating',axis=1)
baseTest['Heating_Floor']=0
baseTest['Heating_OthW']=0

# __________________________________________________________________________________________________

# Tratamento coluna HeatingQC: Heating quality and condition

#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['HeatingQC']])
dfOde=dfOde.rename({'HeatingQC':'HeatingQC_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['HeatingQC_Ode'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('HeatingQC',axis=1)
baseTest=baseTest.drop('HeatingQC_Ode',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna CentralAir: Central air conditioning

#        N	No
#        Y	Yes

baseTest['CentralAir'].value_counts()

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['CentralAir']])
dfOde=dfOde.rename({'CentralAir':'CentralAir_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['CentralAir_Ode'].value_counts()

baseTest=baseTest.drop('CentralAir',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna Electrical: Electrical system

#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed

#Coluna Categórica, one hot Encoding dos dados
# baseTest['Electrical'].value_counts()
# dfOhe=oneHotEncoding(baseTest[['Electrical']])
# baseTest=pd.concat([baseTest,dfOhe],axis=1)
# baseTest=baseTest.drop('Electrical',axis=1)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('Electrical',axis=1)

# __________________________________________________________________________________________________
		
# Tratamento coluna 1stFlrSF: First Floor square feet
 # Sem tratamento necessário
# __________________________________________________________________________________________________
 
# Tratamento coluna 2ndFlrSF: Second floor square feet
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# Tratamento coluna LowQualFinSF: Low quality finished square feet (all floors)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('LowQualFinSF',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna GrLivArea: Above grade (ground) living area square feet
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# Tratamento coluna BsmtFullBath: Basement full bathrooms
baseTest['BsmtFullBath'].value_counts()
baseTest['BsmtFullBath']=baseTest['BsmtFullBath'].fillna(0)

# __________________________________________________________________________________________________

# BsmtHalfBath: Basement half bathrooms
baseTest['BsmtHalfBath'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('BsmtHalfBath',axis=1)

# __________________________________________________________________________________________________

# FullBath: Full bathrooms above grade
baseTest['FullBath'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# HalfBath: Half baths above grade
baseTest['HalfBath'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
baseTest['BedroomAbvGr'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# Kitchen: Kitchens above grade
baseTest['KitchenAbvGr'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# KitchenQual: Kitchen quality

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
baseTest['KitchenQual'].value_counts()
baseTest[baseTest['KitchenQual']==""]
baseTest['KitchenQual']=baseTest['KitchenQual'].fillna('TA')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['KitchenQual']])
dfOde=dfOde.rename({'KitchenQual':'KitchenQual_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['KitchenQual_Ode'].value_counts()

baseTest=baseTest.drop('KitchenQual',axis=1)

# __________________________________________________________________________________________________
       	
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
baseTest['TotRmsAbvGrd'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# Functional: Home functionality (Assume typical unless deductions are warranted)

#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only

baseTest['Functional'].value_counts()
baseTest[baseTest['Functional'].isna()]
baseTest['Functional']=baseTest['Functional'].fillna('Typ')

#Coluna Categórica, one hot Encoding dos dados
baseTest['Functional'].value_counts()
dfOhe=oneHotEncoding(baseTest[['Functional']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('Functional',axis=1)

# __________________________________________________________________________________________________
		
# Fireplaces: Number of fireplaces
baseTest['Fireplaces'].value_counts()
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# FireplaceQu: Fireplace quality

#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
baseTest['FireplaceQu'].value_counts()
baseTest[baseTest['FireplaceQu'].isna()]
baseTest['FireplaceQu']=baseTest['FireplaceQu'].fillna('No Fireplace')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['FireplaceQu']])
dfOde=dfOde.rename({'FireplaceQu':'FireplaceQu_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['FireplaceQu_Ode'].value_counts()
baseTest=baseTest.drop('FireplaceQu',axis=1)


# __________________________________________________________________________________________________
		
# GarageType: Garage location
		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
baseTest['GarageType'].value_counts()
baseTest[baseTest['GarageType'].isna()]
baseTest['GarageType']=baseTest['GarageType'].fillna('No Garage')

#Coluna Categórica, one hot Encoding dos dados
baseTest['GarageType'].value_counts()
dfOhe=oneHotEncoding(baseTest[['GarageType']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('GarageType',axis=1)

# __________________________________________________________________________________________________
		
# GarageYrBlt: Year garage was built

baseTest['GarageYrBlt'].sort_values().unique()
baseTest[baseTest['GarageYrBlt']==""]
baseTest.loc[baseTest['GarageYrBlt']=='','GarageYrBlt']=0

#Função para transformar os dados de ano em categorias por períodos:
def clasAnos(x):
        
    if x >=1800 and x <=1899: 
        return '1800 a 1899'
    elif x >=1900 and x <=1999:
        return '1900 a 1999'
    elif x > 2000:
        return 'acima de 2000'
    else:
        return 'No Garage'
    
baseTest['GarageYrBlt']=baseTest['GarageYrBlt'].apply(clasAnos)

#Fazendo o OneHot Encoding dos dados categoricos criados
dfOhe=oneHotEncoding(baseTest[['GarageYrBlt']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('GarageYrBlt',axis=1)


# __________________________________________________________________________________________________
	
# GarageFinish: Interior finish of the garage

#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage

baseTest['GarageFinish'].value_counts()
baseTest[baseTest['GarageFinish'].isna()]
baseTest['GarageFinish']=baseTest['GarageFinish'].fillna('No Garage')

#Coluna Categórica, one hot Encoding dos dados
baseTest['GarageFinish'].value_counts()
dfOhe=oneHotEncoding(baseTest[['GarageFinish']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('GarageFinish',axis=1)

# __________________________________________________________________________________________________

# GarageCars: Size of garage in car capacity
baseTest['GarageCars'].value_counts()
baseTest[baseTest['GarageCars'].isnull()]
baseTest['GarageCars']=baseTest['GarageCars'].fillna(baseTest['GarageCars'].median())
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# GarageArea: Size of garage in square feet
baseTest['GarageArea'].value_counts()
baseTest[baseTest['GarageCars'].isnull()]
baseTest['GarageArea']=baseTest['GarageArea'].fillna(baseTest['GarageArea'].median())
 # Sem tratamento necessário

# __________________________________________________________________________________________________

# GarageQual: Garage quality

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
baseTest['GarageQual'].value_counts()
baseTest[baseTest['GarageQual'].isna()]
baseTest['GarageQual']=baseTest['GarageQual'].fillna('No Garage')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['GarageQual']])
dfOde=dfOde.rename({'GarageQual':'GarageQual_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['GarageQual_Ode'].value_counts()
baseTest=baseTest.drop('GarageQual',axis=1)

# __________________________________________________________________________________________________
		
# GarageCond: Garage condition

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
baseTest['GarageCond'].value_counts()
baseTest[baseTest['GarageCond'].isna()]
baseTest['GarageCond']=baseTest['GarageCond'].fillna('No Garage')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['GarageCond']])
dfOde=dfOde.rename({'GarageCond':'GarageCond_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['GarageCond_Ode'].value_counts()
baseTest=baseTest.drop('GarageCond',axis=1)

# __________________________________________________________________________________________________
	
# PavedDrive: Paved driveway

#        Y	Paved 
#        P	Partial Pavement
#        N	Dirt/Gravel
baseTest['PavedDrive'].value_counts()

#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['PavedDrive']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('PavedDrive',axis=1)

# __________________________________________________________________________________________________
		
# WoodDeckSF: Wood deck area in square feet
baseTest['WoodDeckSF'].value_counts()

# __________________________________________________________________________________________________

# OpenPorchSF: Open porch area in square feet
baseTest['OpenPorchSF'].value_counts()


# __________________________________________________________________________________________________

# EnclosedPorch: Enclosed porch area in square feet
baseTest['EnclosedPorch'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('EnclosedPorch',axis=1)

# __________________________________________________________________________________________________

# 3SsnPorch: Three season porch area in square feet
baseTest['3SsnPorch'].value_counts()

 # Sem tratamento necessário

# __________________________________________________________________________________________________

# ScreenPorch: Screen porch area in square feet
baseTest['ScreenPorch'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTest=baseTest.drop('ScreenPorch',axis=1)
# __________________________________________________________________________________________________

# PoolArea: Pool area in square feet
baseTest['PoolArea'].value_counts()

 # Sem tratamento necessário


# __________________________________________________________________________________________________

# PoolQC: Pool quality
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
baseTest['PoolQC'].value_counts()

baseTest[baseTest['PoolQC'].isna()]
baseTest['PoolQC']=baseTest['PoolQC'].fillna('No Pool')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['PoolQC']])
dfOde=dfOde.rename({'PoolQC':'PoolQC_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['PoolQC_Ode'].value_counts()
baseTest=baseTest.drop('PoolQC',axis=1)

# __________________________________________________________________________________________________
		
# Fence: Fence quality
		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
baseTest['Fence'].value_counts()

baseTest[baseTest['Fence'].isna()]
baseTest['Fence']=baseTest['Fence'].fillna('No Fence')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTest[['Fence']])
dfOde=dfOde.rename({'Fence':'Fence_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['Fence_Ode'].value_counts()
baseTest=baseTest.drop('Fence',axis=1)

# __________________________________________________________________________________________________

# MiscFeature: Miscellaneous feature not covered in other categories
		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
baseTest['MiscFeature'].value_counts()

#Pouca correlação com a coluna SelePrice e muitos valores NA, não será considerada no modelo
baseTest=baseTest.drop('MiscFeature',axis=1)
# __________________________________________________________________________________________________
	
# MiscVal: $Value of miscellaneous feature
baseTest['MiscVal'].value_counts()
#Pouca correlação com a coluna SelePrice e muitos valores NA, não será considerada no modelo
baseTest=baseTest.drop('MiscVal',axis=1)

# __________________________________________________________________________________________________

# MoSold: Month Sold (MM)
baseTest['MoSold'].sort_values().unique()

 # Sem tratamento necessário

# __________________________________________________________________________________________________

# YrSold: Year Sold (YYYY)
baseTest['YrSold'].sort_values().unique()

dfOde=ordinalEncoding(baseTest[['YrSold']])
dfOde=dfOde.rename({'YrSold':'YrSold_Ode'},axis=1)
baseTest=pd.concat([baseTest,dfOde],axis=1)
baseTest['YrSold_Ode'].value_counts()
baseTest=baseTest.drop('YrSold',axis=1)



# __________________________________________________________________________________________________

# SaleType: Type of sale
		
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
baseTest['SaleType'].value_counts()
baseTest[baseTest['SaleType'].isna()]
baseTest['SaleType']=baseTest['SaleType'].fillna('WD')

#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['SaleType']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('SaleType',axis=1)

# __________________________________________________________________________________________________
		
# SaleCondition: Condition of sale

#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
baseTest['SaleCondition'].value_counts()
#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTest[['SaleCondition']])
baseTest=pd.concat([baseTest,dfOhe],axis=1)
baseTest=baseTest.drop('SaleCondition',axis=1)

baseTest.to_excel('baseTestTratada.xlsx',index=False)

# Calcula a matriz de correlação
correlation_matrix = baseTest.corr()

# Exibe a matriz de correlação
# correlation_matrix.to_excel('correlation_matrixTest.xlsx')


