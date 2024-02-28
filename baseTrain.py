import pandas as pd
import polars as pl
import lux
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#Análise exploratória dos dados e encoding

baseTrain=pd.read_csv('train.csv',sep=",")

len(baseTrain.columns)

baseTrain.info()

baseTrain.head()

baseTrain.describe()

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

baseTrain['MSSubClass']=baseTrain['MSSubClass'].replace(categories)

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['MSSubClass']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('MSSubClass',axis=1)
baseTrain['MSSubClass_1-1/2 STORY PUD - ALL AGES']=0

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

baseTrain['MSZoning'].value_counts()

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['MSZoning']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('MSZoning',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotFrontage:
#Valores Nulos substituidos pela mediana
baseTrain['LotFrontage']=baseTrain['LotFrontage'].fillna(baseTrain['LotFrontage'].median())

#__________________________________________________________________________________________________

# Tratamento coluna Street:
#Colua categórica com dua opções, encoding para 0 e 1
baseTrain['Street'].value_counts()

baseTrain.query("Street == 'Grvl'")

baseTrain.query("Street == 'Pave'")

baseTrain['Street']=baseTrain['Street'].apply(lambda x: 0 if x=='Pave' else 1)

#__________________________________________________________________________________________________

# Tratamento coluna Alley:
#Maioria dos valores nulo, não será considerada no modelo
baseTrain['Alley'].value_counts()
baseTrain=baseTrain.drop('Alley',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotShape:
# LotShape: General shape of property

#        Reg	Regular	
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['LotShape']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('LotShape',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LandContour:
# LandContour: Flatness of the property

#        Lvl	Near Flat/Level	
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['LandContour']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('LandContour',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Utilities:
#Maioria dos valores é AllPub, não será considerada no modelo
baseTrain=baseTrain.drop('Utilities',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LotConfig:
# LotConfig: Lot configuration

#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['LotConfig']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('LotConfig',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna LandSlope:
# LandSlope: Slope of property
		
#        Gtl	Gentle slope
#        Mod	Moderate Slope	
#        Sev	Severe Slope
       
#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['LandSlope']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('LandSlope',axis=1)

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
dfOhe=oneHotEncoding(baseTrain[['Neighborhood']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Neighborhood',axis=1)

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
dfOhe=oneHotEncoding(baseTrain[['Condition1']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Condition1',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Condition2:
#Similar a coluna condition1, não será considerada no modelo
baseTrain=baseTrain.drop('Condition2',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BldgType:
# BldgType: Type of dwelling
		
#        1Fam	Single-family Detached	
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['BldgType']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('BldgType',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna HouseStyle:
#Similar a coluna MSSubClass, não será considerada no modelo
baseTrain=baseTrain.drop('HouseStyle',axis=1)


#__________________________________________________________________________________________________

# Tratamento colunas YearBuilt e YearRemodAdd

baseTrain['YearBuilt'].sort_values().unique()
baseTrain['YearRemodAdd'].sort_values().unique()

#Função para transformar os dados de ano em categorias por períodos:
def clasAnos(x):
        
    if x >=1800 and x <=1899: 
        return '1800 a 1899'
    elif x >=1900 and x <=1999:
        return '1900 a 1900'
    else:
        return 'acima de 2000'
        
baseTrain['YearBuilt']=baseTrain['YearBuilt'].apply(clasAnos)
baseTrain['YearRemodAdd']=baseTrain['YearRemodAdd'].apply(clasAnos)

#Fazendo o OneHot Encoding dos dados categoricos criados
dfOhe=oneHotEncoding(baseTrain[['YearBuilt']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('YearBuilt',axis=1)
dfOhe=oneHotEncoding(baseTrain[['YearRemodAdd']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('YearRemodAdd',axis=1)

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
dfOhe=oneHotEncoding(baseTrain[['RoofStyle']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('RoofStyle',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna RoofMatl
baseTrain['RoofMatl'].value_counts()
#Similar a coluna RoofStyle, não será considerada no modelo
baseTrain=baseTrain.drop('RoofMatl',axis=1)

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
baseTrain[baseTrain['Exterior1st']==" "]

baseTrain.iloc[691,:10]

dfOhe=oneHotEncoding(baseTrain[['Exterior1st']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Exterior1st',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Exterior2nd
#Similar a coluna Exterior1st, não será considerada no modelo
baseTrain=baseTrain.drop('Exterior2nd',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna MasVnrType
baseTrain['MasVnrType'].value_counts()
baseTrain['MasVnrType']=baseTrain['MasVnrType'].fillna('não tem')
baseTrain.loc[baseTrain['MasVnrType'].isnull()]

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['MasVnrType']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('MasVnrType',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna MasVnrArea
baseTrain['MasVnrArea'].value_counts()
baseTrain.loc[baseTrain['MasVnrArea'].isnull()]
baseTrain['MasVnrArea']=baseTrain['MasVnrArea'].fillna(0)

#__________________________________________________________________________________________________
 
# Tratamento coluna ExterQual
# ExterQual: Evaluates the quality of the material on the exterior 
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['ExterQual']])
dfOde=dfOde.rename({'ExterQual':'ExterQual_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain=baseTrain.drop('ExterQual',axis=1)

# Tratamento coluna ExterCond
baseTrain['ExterCond'].value_counts()
# ExterCond: Evaluates the present condition of the material on the exterior
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['ExterCond']])
dfOde=dfOde.rename({'ExterCond':'ExterCond_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain=baseTrain.drop('ExterCond',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna Foundation
baseTrain['Foundation'].value_counts()
# Foundation: Type of foundation
		
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete	
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood

#Coluna Categórica, OneHot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['Foundation']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Foundation',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtQual
baseTrain['BsmtQual'].value_counts()
baseTrain['BsmtQual']=baseTrain['BsmtQual'].fillna("No Basement")
# BsmtQual: Evaluates the height of the basement

#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['BsmtQual']])
dfOde=dfOde.rename({'BsmtQual':'BsmtQual_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain=baseTrain.drop('BsmtQual',axis=1)

baseTrain['BsmtQual_Ode'].value_counts()

#__________________________________________________________________________________________________

# Tratamento coluna BsmtCond
baseTrain['BsmtCond'].value_counts()
baseTrain['BsmtCond']=baseTrain['BsmtCond'].fillna("No Basement")
# BsmtCond: Evaluates the general condition of the basement

#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['BsmtCond']])
dfOde=dfOde.rename({'BsmtCond':'BsmtCond_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain=baseTrain.drop('BsmtCond',axis=1)

correlation_matrix = baseTrain[['BsmtCond_Ode', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtExposure
baseTrain['BsmtExposure'].value_counts()
baseTrain['BsmtExposure']=baseTrain['BsmtExposure'].fillna("No Basement")
baseTrain.loc[baseTrain['BsmtExposure']=='No','BsmtExposure']='No Exposure'

# BsmtExposure: Refers to walkout or garden level walls

#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement

dfOde=ordinalEncoding(baseTrain[['BsmtExposure']])
dfOde=dfOde.rename({'BsmtExposure':'BsmtExposure_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)

correlation_matrix = baseTrain[['BsmtExposure_Ode', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtExposure',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtFinType1: Rating of basement finished area
baseTrain['BsmtFinType1'].value_counts()
baseTrain[baseTrain['BsmtFinType1'].isnull()]
baseTrain['BsmtFinType1']=baseTrain['BsmtFinType1'].fillna("No Basement")

    #    GLQ	Good Living Quarters
    #    ALQ	Average Living Quarters
    #    BLQ	Below Average Living Quarters	
    #    Rec	Average Rec Room
    #    LwQ	Low Quality
    #    Unf	Unfinshed
    #    NA	No Basement
    
#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['BsmtFinType1']])
dfOde=dfOde.rename({'BsmtFinType1':'BsmtFinType1_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['BsmtFinType1_Ode'].value_counts()

correlation_matrix = baseTrain[['BsmtFinType1_Ode', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtFinType1',axis=1)
baseTrain=baseTrain.drop('BsmtFinType1_Ode',axis=1)

#__________________________________________________________________________________________________
# Tratamento coluna BsmtFinSF1: Type 1 finished square feet
correlation_matrix = baseTrain[['BsmtFinSF1', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtFinSF1',axis=1)

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
baseTrain=baseTrain.drop('BsmtFinType2',axis=1)

#__________________________________________________________________________________________________

# Tratamento coluna BsmtFinSF2: Type 2 finished square feet

baseTrain['BsmtFinSF2'].value_counts()

correlation_matrix = baseTrain[['BsmtFinSF2', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtFinSF2',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna BsmtUnfSF: Unfinished square feet of basement area
correlation_matrix = baseTrain[['BsmtUnfSF', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtUnfSF',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna TotalBsmtSF: Total square feet of basement area

correlation_matrix = baseTrain[['TotalBsmtSF', 'SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
baseTrain['TotalBsmtSF'].value_counts()

# __________________________________________________________________________________________________

# Tratamento coluna Heating: Type of heating
		
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace	
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace

baseTrain['Heating'].value_counts()
dfOhe=oneHotEncoding(baseTrain[['Heating']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Heating',axis=1)

correlation_matrix = baseTrain[['Heating_Floor','Heating_GasA','Heating_GasW','Heating_Grav','SalePrice','Heating_OthW','Heating_Wall']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# Tratamento coluna HeatingQC: Heating quality and condition

#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['HeatingQC']])
dfOde=dfOde.rename({'HeatingQC':'HeatingQC_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['HeatingQC_Ode'].value_counts()

correlation_matrix = baseTrain[['HeatingQC_Ode','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('HeatingQC',axis=1)
baseTrain=baseTrain.drop('HeatingQC_Ode',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna CentralAir: Central air conditioning

#        N	No
#        Y	Yes

baseTrain['CentralAir'].value_counts()

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['CentralAir']])
dfOde=dfOde.rename({'CentralAir':'CentralAir_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['CentralAir_Ode'].value_counts()

correlation_matrix = baseTrain[['CentralAir_Ode','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
baseTrain=baseTrain.drop('CentralAir',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna Electrical: Electrical system

#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed

#Coluna Categórica, one hot Encoding dos dados
baseTrain['Electrical'].value_counts()
# dfOhe=oneHotEncoding(baseTrain[['Electrical']])
# baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
# baseTrain=baseTrain.drop('Electrical',axis=1)

# correlation_matrix = baseTrain[['Electrical_FuseA','Electrical_FuseF','Electrical_FuseP','Electrical_Mix','SalePrice']].corr()
# # Exibir a matriz de correlação
# print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop(['Electrical'],axis=1)

# __________________________________________________________________________________________________
		
# Tratamento coluna 1stFlrSF: First Floor square feet
correlation_matrix = baseTrain[['1stFlrSF','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)


# __________________________________________________________________________________________________
 
# Tratamento coluna 2ndFlrSF: Second floor square feet
correlation_matrix = baseTrain[['2ndFlrSF','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# Tratamento coluna LowQualFinSF: Low quality finished square feet (all floors)
correlation_matrix = baseTrain[['LowQualFinSF','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('LowQualFinSF',axis=1)

# __________________________________________________________________________________________________

# Tratamento coluna GrLivArea: Above grade (ground) living area square feet
correlation_matrix = baseTrain[['GrLivArea','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# Tratamento coluna BsmtFullBath: Basement full bathrooms
baseTrain['BsmtFullBath'].value_counts()
baseTrain['BsmtFullBath']=baseTrain['BsmtFullBath'].fillna(0)
correlation_matrix = baseTrain[['BsmtFullBath','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# BsmtHalfBath: Basement half bathrooms
baseTrain['BsmtHalfBath'].value_counts()
correlation_matrix = baseTrain[['BsmtHalfBath','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('BsmtHalfBath',axis=1)

# __________________________________________________________________________________________________

# FullBath: Full bathrooms above grade
baseTrain['FullBath'].value_counts()
correlation_matrix = baseTrain[['FullBath','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# HalfBath: Half baths above grade
baseTrain['HalfBath'].value_counts()
correlation_matrix = baseTrain[['HalfBath','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
baseTrain['BedroomAbvGr'].value_counts()
correlation_matrix = baseTrain[['BedroomAbvGr','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# Kitchen: Kitchens above grade
baseTrain['KitchenAbvGr'].value_counts()
correlation_matrix = baseTrain[['KitchenAbvGr','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# KitchenQual: Kitchen quality

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
baseTrain['KitchenQual'].value_counts()
baseTrain[baseTrain['KitchenQual']==""]

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['KitchenQual']])
dfOde=dfOde.rename({'KitchenQual':'KitchenQual_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['KitchenQual_Ode'].value_counts()

baseTrain=baseTrain.drop('KitchenQual',axis=1)

correlation_matrix = baseTrain[['KitchenQual_Ode','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
# __________________________________________________________________________________________________
       	
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
baseTrain['TotRmsAbvGrd'].value_counts()
correlation_matrix = baseTrain[['TotRmsAbvGrd','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

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

baseTrain['Functional'].value_counts()
baseTrain[baseTrain['Functional'].isna()]

#Coluna Categórica, one hot Encoding dos dados
baseTrain['Functional'].value_counts()
dfOhe=oneHotEncoding(baseTrain[['Functional']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('Functional',axis=1)

# __________________________________________________________________________________________________
		
# Fireplaces: Number of fireplaces
baseTrain['Fireplaces'].value_counts()
correlation_matrix = baseTrain[['Fireplaces','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# FireplaceQu: Fireplace quality

#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
baseTrain['FireplaceQu'].value_counts()
baseTrain[baseTrain['FireplaceQu'].isna()]
baseTrain['FireplaceQu']=baseTrain['FireplaceQu'].fillna('No Fireplace')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['FireplaceQu']])
dfOde=dfOde.rename({'FireplaceQu':'FireplaceQu_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['FireplaceQu_Ode'].value_counts()
baseTrain=baseTrain.drop('FireplaceQu',axis=1)


# __________________________________________________________________________________________________
		
# GarageType: Garage location
		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
baseTrain['GarageType'].value_counts()
baseTrain[baseTrain['GarageType'].isna()]
baseTrain['GarageType']=baseTrain['GarageType'].fillna('No Garage')

#Coluna Categórica, one hot Encoding dos dados
baseTrain['GarageType'].value_counts()
dfOhe=oneHotEncoding(baseTrain[['GarageType']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('GarageType',axis=1)

# __________________________________________________________________________________________________
		
# GarageYrBlt: Year garage was built

baseTrain['GarageYrBlt'].sort_values().unique()
baseTrain[baseTrain['GarageYrBlt']==""]
baseTrain.loc[baseTrain['GarageYrBlt']=='','GarageYrBlt']=0

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
    
baseTrain['GarageYrBlt']=baseTrain['GarageYrBlt'].apply(clasAnos)

#Fazendo o OneHot Encoding dos dados categoricos criados
dfOhe=oneHotEncoding(baseTrain[['GarageYrBlt']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('GarageYrBlt',axis=1)
baseTrain['GarageYrBlt_1800 a 1899']=0

# __________________________________________________________________________________________________
	
# GarageFinish: Interior finish of the garage

#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage

baseTrain['GarageFinish'].value_counts()
baseTrain[baseTrain['GarageFinish'].isna()]
baseTrain['GarageFinish']=baseTrain['GarageFinish'].fillna('No Garage')

#Coluna Categórica, one hot Encoding dos dados
baseTrain['GarageFinish'].value_counts()
dfOhe=oneHotEncoding(baseTrain[['GarageFinish']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('GarageFinish',axis=1)

# __________________________________________________________________________________________________

# GarageCars: Size of garage in car capacity
baseTrain['GarageCars'].value_counts()
baseTrain[baseTrain['GarageCars'].isnull()]

correlation_matrix = baseTrain[['GarageCars','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
# __________________________________________________________________________________________________

# GarageArea: Size of garage in square feet
baseTrain['GarageArea'].value_counts()
baseTrain[baseTrain['GarageCars'].isnull()]

correlation_matrix = baseTrain[['GarageArea','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
# __________________________________________________________________________________________________

# GarageQual: Garage quality

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
baseTrain['GarageQual'].value_counts()
baseTrain[baseTrain['GarageQual'].isna()]
baseTrain['GarageQual']=baseTrain['GarageQual'].fillna('No Garage')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['GarageQual']])
dfOde=dfOde.rename({'GarageQual':'GarageQual_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['GarageQual_Ode'].value_counts()
baseTrain=baseTrain.drop('GarageQual',axis=1)

# __________________________________________________________________________________________________
		
# GarageCond: Garage condition

#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
baseTrain['GarageCond'].value_counts()
baseTrain[baseTrain['GarageCond'].isna()]
baseTrain['GarageCond']=baseTrain['GarageCond'].fillna('No Garage')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['GarageCond']])
dfOde=dfOde.rename({'GarageCond':'GarageCond_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['GarageCond_Ode'].value_counts()
baseTrain=baseTrain.drop('GarageCond',axis=1)

# __________________________________________________________________________________________________
	
# PavedDrive: Paved driveway

#        Y	Paved 
#        P	Partial Pavement
#        N	Dirt/Gravel
baseTrain['PavedDrive'].value_counts()

#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['PavedDrive']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('PavedDrive',axis=1)

# __________________________________________________________________________________________________
		
# WoodDeckSF: Wood deck area in square feet
baseTrain['WoodDeckSF'].value_counts()

correlation_matrix = baseTrain[['WoodDeckSF','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)
# __________________________________________________________________________________________________

# OpenPorchSF: Open porch area in square feet
baseTrain['OpenPorchSF'].value_counts()

correlation_matrix = baseTrain[['OpenPorchSF','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# EnclosedPorch: Enclosed porch area in square feet
baseTrain['EnclosedPorch'].value_counts()

correlation_matrix = baseTrain[['EnclosedPorch','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('EnclosedPorch',axis=1)

# __________________________________________________________________________________________________

# 3SsnPorch: Three season porch area in square feet
baseTrain['3SsnPorch'].value_counts()

correlation_matrix = baseTrain[['3SsnPorch','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# ScreenPorch: Screen porch area in square feet
baseTrain['ScreenPorch'].value_counts()

#Pouca correlação com a coluna SelePrice, não será considerada no modelo
baseTrain=baseTrain.drop('ScreenPorch',axis=1)
# __________________________________________________________________________________________________

# PoolArea: Pool area in square feet
baseTrain['PoolArea'].value_counts()

correlation_matrix = baseTrain[['PoolArea','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# PoolQC: Pool quality
		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
baseTrain['PoolQC'].value_counts()

baseTrain[baseTrain['PoolQC'].isna()]
baseTrain['PoolQC']=baseTrain['PoolQC'].fillna('No Pool')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['PoolQC']])
dfOde=dfOde.rename({'PoolQC':'PoolQC_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['PoolQC_Ode'].value_counts()
baseTrain=baseTrain.drop('PoolQC',axis=1)

# __________________________________________________________________________________________________
		
# Fence: Fence quality
		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
baseTrain['Fence'].value_counts()

baseTrain[baseTrain['Fence'].isna()]
baseTrain['Fence']=baseTrain['Fence'].fillna('No Fence')

#Coluna Categórica com ordem intrínsica, Ordinal Encoding dos dados
dfOde=ordinalEncoding(baseTrain[['Fence']])
dfOde=dfOde.rename({'Fence':'Fence_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['Fence_Ode'].value_counts()
baseTrain=baseTrain.drop('Fence',axis=1)

# __________________________________________________________________________________________________

# MiscFeature: Miscellaneous feature not covered in other categories
		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
baseTrain['MiscFeature'].value_counts()

#Pouca correlação com a coluna SelePrice e muitos valores NA, não será considerada no modelo
baseTrain=baseTrain.drop('MiscFeature',axis=1)
# __________________________________________________________________________________________________
	
# MiscVal: $Value of miscellaneous feature
baseTrain['MiscVal'].value_counts()
#Pouca correlação com a coluna SelePrice e muitos valores NA, não será considerada no modelo
baseTrain=baseTrain.drop('MiscVal',axis=1)

# __________________________________________________________________________________________________

# MoSold: Month Sold (MM)
baseTrain['MoSold'].sort_values().unique()

correlation_matrix = baseTrain[['MoSold','SalePrice']].corr()
# Exibir a matriz de correlação
print(correlation_matrix)

# __________________________________________________________________________________________________

# YrSold: Year Sold (YYYY)
baseTrain['YrSold'].sort_values().unique()

dfOde=ordinalEncoding(baseTrain[['YrSold']])
dfOde=dfOde.rename({'YrSold':'YrSold_Ode'},axis=1)
baseTrain=pd.concat([baseTrain,dfOde],axis=1)
baseTrain['YrSold_Ode'].value_counts()
baseTrain=baseTrain.drop('YrSold',axis=1)



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
baseTrain['SaleType'].value_counts()

#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['SaleType']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('SaleType',axis=1)

# __________________________________________________________________________________________________
		
# SaleCondition: Condition of sale

#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
baseTrain['SaleCondition'].value_counts()
#Coluna Categórica, one hot Encoding dos dados
dfOhe=oneHotEncoding(baseTrain[['SaleCondition']])
baseTrain=pd.concat([baseTrain,dfOhe],axis=1)
baseTrain=baseTrain.drop('SaleCondition',axis=1)

# Obter lista de colunas excluindo a coluna alvo
cols = [col for col in baseTrain.columns if col != 'SalePrice']

# Reorganizar as colunas (mover a coluna alvo para o final)
baseTrain = baseTrain[cols + ['SalePrice']]

baseTrain.to_excel('baseTreinoTratada.xlsx',index=False)

# Calcula a matriz de correlação
correlation_matrix = baseTrain.corr()

# Exibe a matriz de correlação
correlation_matrix.to_excel('correlation_matrix.xlsx')

