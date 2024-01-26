import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.")
warnings.filterwarnings("ignore", message="Calling float on a single element Series is deprecated.*")

class DCF:
  def __init__(self, stock, key):
    self.key = key
    self.stock = stock
    self.fd = FundamentalData(key = self.key, output_format = 'pandas')
    self.ts = TimeSeries(key = self.key, output_format = 'pandas')
    self.incomeStatement = pd.DataFrame()
    self.balanceSheet = pd.DataFrame()
    self.cashFlow = pd.DataFrame()
    self.overview = pd.DataFrame()
    self.yearsArr = np.array([])
    self.marketRiskPremium = 0.05
    self.terminalGrowthRate = 0.02
    self.beta = 0.0
    self.evebitda = 0.0
    self.initialGrowthRate = 0.0
    self.riskFreeRate = 0.0

  # call this something differnt but only CALL it ONCE
  def retrieveIS(self):
    alphaIncomeStatement = self.fd.get_income_statement_annual(self.stock)
    self.incomeStatement = alphaIncomeStatement[0][:5].T.replace('None', 0)

  def retrieveBS(self):
    alphaBalanceSheet = self.fd.get_balance_sheet_annual(self.stock)
    self.balanceSheet = alphaBalanceSheet[0][:5].T.replace('None', 0)

  def getIncomeStatement(self):
    if self.incomeStatement.empty:
      self.retrieveIS()
    return self.incomeStatement

  def getBalanceSheet(self):
    if self.balanceSheet.empty:
      self.retrieveBS()
    return self.balanceSheet

  def retrieveCF(self):
    alphaCashFlow = self.fd.get_cash_flow_annual(self.stock)
    self.cashFlow = alphaCashFlow[0][:5].T.replace('None', 0)

  def getCashFlow(self):
    if self.cashFlow.empty:
      self.retrieveCF()
    return self.cashFlow

  def retrieveOverview(self):
    ticker = yf.Ticker(self.stock)
    self.overview = pd.DataFrame(ticker.info).iloc[0]

  def getOverview(self):
    if self.overview.empty:
      self.retrieveOverview()
    return self.overview

  def getYears(self):
    if np.size(self.yearsArr) == 0:
      fiscalYear = []
      for i in reversed(range(0,5)):
          fiscalYear.append(int(self.getIncomeStatement().loc['fiscalDateEnding'][i][0:4]))
      for i in range(4,9):
          fiscalYear.append(fiscalYear[i]+1)
      self.yearsArr = np.array(fiscalYear)
    return self.yearsArr

  def get5YearDataIS(self, stri):
    inc_stat = self.getIncomeStatement()
    return np.array(inc_stat.iloc[:, ::-1].loc[stri]).astype('int64')

  def get5YearGrowth(self, stri):
    arr = self.get5YearDataIS(stri)
    growth = [0.0]
    for i in range(1, len(arr)):
      growth.append((arr[i]-arr[i-1])/arr[i-1])
    return np.array(growth)

  def get5YearPercentOfRevenue(self, stri):
    arr = self.get5YearDataIS(stri)
    revenue = self.get5YearDataIS('totalRevenue')
    return (arr/revenue)

  def getAvg5YearGrowth(self, stri):
    arr = self.get5YearGrowth(stri)
    arr = np.delete(arr, 0)
    return arr.mean()

  def getAvg5YearPercentOfRevenue(self, stri):
    arr = self.get5YearPercentOfRevenue(stri)
    arr = np.delete(arr, 0)
    return arr.mean()

  def tenYearGrowthRate(self, stri):
    if self.initialGrowthRate == 0.0:
      print("The average growth rate is: " + str(self.getAvg5YearGrowth(stri)))
      self.initialGrowthRate = float(input("Input initial year Growth Rate for " + self.stock + ": "))

    arrGrowth = np.append(self.get5YearGrowth(stri), self.initialGrowthRate)
    multiplier = .9
    for i in range(6, 10):
      arrGrowth = np.append(arrGrowth, arrGrowth[i-1]*multiplier)
      multiplier *= 0.9

    return arrGrowth

  def projectRevenue(self):
    arrGrowth = self.tenYearGrowthRate('totalRevenue')
    arrRevenue = self.get5YearDataIS('totalRevenue')
    for i in range(5, 10):
      arrRevenue = np.append(arrRevenue, arrRevenue[i-1]*(1+arrGrowth[i]))

    return arrRevenue

  def tenYearPercentOfRev(self, stri):
    arrPercent = self.get5YearPercentOfRevenue(stri)
    avgPercent = self.getAvg5YearPercentOfRevenue(stri)
    self.get5YearDataIS('totalRevenue')
    self.get5YearDataIS(stri)
    for _i in range(5, 10):
      arrPercent = np.append(arrPercent, avgPercent)
    return arrPercent

  def calcPercentOfRev(self, nparr):
    arrRevenue = self.projectRevenue()
    arrPercent = []
    for i in range (0,10):
      arrPercent = np.append(arrPercent, nparr[i] / arrRevenue[i])
    return arrPercent

  def projectExpense(self, stri):
    arrPercent = self.tenYearPercentOfRev(stri)
    revenue = self.projectRevenue()
    data = self.get5YearDataIS(stri)
    for i in range(5, 10):
      data = np.append(data,revenue[i] * arrPercent[i])
    return data

  def projectCOGS(self):
    return self.projectExpense('costofGoodsAndServicesSold')

  def projectGrossProfit(self):
    return self.projectRevenue() - self.projectCOGS()

  def projectSGA(self):
    return self.projectExpense('sellingGeneralAndAdministrative')

  def projectDA(self):
    return self.projectExpense('depreciationAndAmortization')

  def projectRD(self):
    return self.projectExpense('researchAndDevelopment')

  def projectOperatingExpenses(self):
    return self.projectExpense('operatingExpenses')

  # calculate operating income
  def projectEBIT(self):
    ebit = self.projectGrossProfit() - self.projectOperatingExpenses()
    return ebit
  def projectNonOperatingExpenses(self):
    # First 5 years: -(Net Income - Operating Income)
    # THEN project percent of revenue of NonOperating Expenses
    # THEN project next 5 years expenses
    # return new array
    netIncome = self.get5YearDataIS('netIncome')
    operatingIncome = self.get5YearDataIS('totalRevenue') - self.get5YearDataIS('costofGoodsAndServicesSold') - self.get5YearDataIS('operatingExpenses')
    nonOperatingExpenses = operatingIncome - netIncome
    revenue = self.get5YearDataIS('totalRevenue')
    percentOfRevenue = nonOperatingExpenses / revenue
    avgPercentOfRevenue = percentOfRevenue.mean()
    revenue = self.projectRevenue()
    for _i in range(5, 10):
      percentOfRevenue = np.append(percentOfRevenue, avgPercentOfRevenue)
    for i in range(5, 10):
      nonOperatingExpenses = np.append(nonOperatingExpenses, percentOfRevenue[i] * revenue[i])
    return nonOperatingExpenses

  def projectNetIncome(self):
    netIncome = self.projectGrossProfit() - self.projectOperatingExpenses() - self.projectNonOperatingExpenses()
    return netIncome

  # RETRIEVE DATA FROM CASH FLOW STATEMENT
  def get5YearDataCF(self, stri):
    cashFlow = self.getCashFlow()
    return np.array(cashFlow.iloc[:, ::-1].loc[stri]).astype('int64')

  # create Revenue Build DataFrame
  def createRevenueDF(self):
    projections = pd.DataFrame()
    projections.index = self.getYears()
    projections.rename_axis('USD in Thousands', axis = 0, inplace = True)
    projections['Revenue'] = (self.projectRevenue()/1000).astype(int)
    projections['Revenue % Growth'] = np.char.mod('%.2f%%', self.tenYearGrowthRate('totalRevenue')*100)
    projections['COGS'] = (self.projectCOGS()/1000).astype(int)
    projections['COGS % of Rev'] = np.char.mod('%.2f%%', self.tenYearPercentOfRev('costofGoodsAndServicesSold')*100)
    projections['Gross Profit'] = (self.projectGrossProfit()/1000).astype(int)
    projections['Operating Exp'] = (self.projectOperatingExpenses()/1000).astype(int)
    projections['Operating Exp % of Rev'] = np.char.mod('%.2f%%', self.tenYearPercentOfRev('operatingExpenses')*100)
    projections['Income From Operations'] = (self.projectEBIT()/1000).astype(int)
    projections['Non-Operating Exp'] = (self.projectNonOperatingExpenses()/1000).astype(int)
    projections['Non-Operating Exp % of Rev'] = np.char.mod('%.2f%%', self.calcPercentOfRev(self.projectNonOperatingExpenses())*100)
    projections['Net Income'] = (self.projectNetIncome()/1000).astype(int)
    return projections.T

  # create Cash Flow Adjustments DataFrame
  def createCFA(self):
    cfa = pd.DataFrame()
    cfa.index = self.getYears()
    cfa['CapEx'] = (self.projectCapEx()/1000).astype(int)
    cfa['CapEx % of Rev'] = np.char.mod('%.2f%%', self.projectPercentOfRevenueCapEx()*100)
    cfa['D&A'] = (self.projectDepAmort()/1000).astype(int)
    cfa['D&A % of CapEx'] = np.char.mod('%.2f%%', self.projectPercentOfCapEx()*100)
    cfa['Total D&A'] = (self.projectDepAmort()/1000).astype(int)
    return cfa.T

  def createNWC(self):
    nwc = pd.DataFrame()
    nwc.index = self.getYears()
    nwc.rename_axis('USD in Thousands', axis = 0, inplace = True)
    nwc['Accounts Receivable'] = (self.projectAR()/1000).astype(int)
    nwc['DSO'] = np.char.mod('%0.2f', self.projectDSO())
    nwc['Inventory'] = (self.projectInv()/1000).astype(int)
    nwc['DIO'] = np.char.mod('%0.2f', self.projectDIO())
    nwc['Prepaid Expenses & Other'] = (self.projectOtherAssets()/1000).astype(int)
    nwc['% of SG&A'] = np.char.mod('%.2f%%', self.projectPercentOfSGA()*100)
    nwc['Total Current Assets'] = (self.projectTotalCurrentAssets()/1000).astype(int)
    nwc['Accounts Payable'] = (self.projectAP()/1000).astype(int)
    nwc['DPO'] = np.char.mod('%0.2f', self.projectDPO())
    nwc['Accrued Liabilities'] = (self.projectAL()/1000).astype(int)
    nwc['% of Sales'] = np.char.mod('%.2f%%', self.projectPercentOfRevenueAL()*100)
    nwc['Total Current Liabilities'] = (self.projectTotalCurrentLiabilities()/1000).astype(int)
    nwc['Net Working Capital'] = (self.projectNWC()/1000).astype(int)
    nwc['Change in NWC'] = (self.projectChangeNWC()/1000).astype(int)
    return nwc.T

  # create FCF DataFrame
  def createFCF(self):
    fcf = pd.DataFrame()
    fcf.index = self.getYears()[5:]
    fcf.rename_axis('USD in Thousands', axis = 0, inplace = True)
    fcf['Revenue'] = (self.projectRevenue()[5:]/1000).astype(int)
    fcf['Rev % Growth'] = np.char.mod('%.2f%%', self.tenYearGrowthRate('totalRevenue')[5:]*100)
    fcf['EBIT'] = (self.projectEBIT()[5:]/1000).astype(int)
    fcf['EBIT Margin'] = np.char.mod('%.2f%%', self.projectEBITMargin()[5:]*100)
    fcf['NOPAT'] = (self.projectNOPAT()[5:]/1000).astype(int)
    fcf['Tax Rate'] = np.char.mod('%.2f%%', self.projectTaxRate()[5:]*100)
    fcf['Plus: D&A'] = (self.projectDepAmort()[5:]/1000).astype(int)
    fcf['Less: CapEX'] = (self.projectCapEx()[5:]/1000).astype(int)
    fcf['Less: Change in NWC'] = (self.projectChangeNWC()[5:]/1000).astype(int)
    fcf['FCFF'] = (self.projectFCFF()[5:]/1000).astype(int)
    fcf['FCFF % Growth'] = np.char.mod('%.2f', self.projectFCFFPercentGrowth()[5:]*100)
    fcf['Discount Period'] = np.char.mod('%.2f', self.projectDiscountPeriod()[5:])
    fcf['Discount Factor'] = np.char.mod('%.2f',self.projectDiscountFactor()[5:])
    fcf['PV of Cash Flows'] = (self.projectPVCashFlow()[5:]/1000).astype(int)
    return fcf.T

  # create Gordon Growth Method DataFrame
  def createGordonGrowth(self):
    gordon = pd.DataFrame()
    gordon.index = ['Info']
    gordon.rename_axis('USD in Thousands', axis = 0, inplace = True)
    gordon['Terminal CF'] = int(self.getTerminalCF()/1000)
    gordon['WACC'] = np.char.mod('%.2f%%', self.getWACC())
    gordon['Terminal Growth Rate'] = np.char.mod('%.2f%%', self.getTerminalGrowthRate())
    gordon['Terminal Value'] = int(self.getTerminalValue()/1000)
    gordon['PV of Terminal Value'] = int(self.getPVOfTerminalValue()/1000)
    gordon['Enterprise Value'] = int(self.getEnterpriseValue()/1000)
    gordon['Less: Debt'] = int(self.getLongDebt()/1000)
    gordon['Plus: Cash'] = int(self.getCash()/1000)
    gordon['Equity Value'] = int(self.getEquityValueGG()/1000)
    gordon['Diluted Shares Oustanding'] = int(self.getSharesOustanding()/1000)
    gordon['Price Per Share'] = np.char.mod('$%.2f', self.getPricePerShareGG())
    return gordon.T

  # create Multiples Method DataFrame
  def createMultiples(self):
    multiples = pd.DataFrame()
    multiples.index = ['Info']
    multiples.rename_axis('USD in Thousands', axis = 0, inplace = True)
    multiples['Terminal EBITDA'] = int(self.getTerminalEBITDA()/1000)
    multiples['EV/EBITDA Multiple'] = np.char.mod('%.2fx', self.getEVEBITDAMultiple())
    multiples['Terminal Value'] = int(self.getTerminalValueMM()/1000)
    multiples['PV of Terminal Value'] = int(self.getPVOfTerminalValueMM()/1000)
    multiples['Enterprise Value'] = int(self.getEnterpriseValueMM()/1000)
    multiples['Less: Debt'] = int(self.getLongDebt()/1000)
    multiples['Plus: Cash'] = int(self.getCash()/1000)
    multiples['Equity Value'] = int(self.getEquityValueMM()/1000)
    multiples['Diluted Shares Oustanding'] = int(self.getSharesOustanding()/1000)
    multiples['Price Per Share'] = np.char.mod('$%.2f', self.getPricePerShareMM())
    return multiples.T

  # create WACC calculation DataFrame
  def createWACC(self):
    wacc = pd.DataFrame()
    wacc.index = [['Info']]
    wacc.rename_axis('USD in Thousands', axis = 0, inplace = True)
    wacc['Market Risk Premium'] = np.char.mod('%.2f%%', self.getMarketRiskPremium()*100)
    wacc['Risk Free Rate'] = np.char.mod('%.2f%%', self.getRiskFreeRate()*100)
    wacc['Beta'] = self.getBeta()
    wacc['Cost of Equity'] = np.char.mod('%.2f%%', self.getCostOfEquity()*100)
    wacc['Pre-Tax Cost of Debt'] = np.char.mod('%.2f%%', self.getPreTaxCostOfDebt()*100)
    wacc['Tax Rate'] = np.char.mod('%.2f%%', self.getTaxRate()*100)
    wacc['After-Tax Cost of Debt'] = np.char.mod('%.2f%%', self.getAfterTaxCostOfDebt()*100)
    wacc['Stock Price'] = np.char.mod('$%.2f', self.getCurrentPrice())
    wacc['Shares Oustanding'] = int(self.getSharesOustanding()/1000)
    wacc['Equity'] = int(self.getCurrentEquityValue()/1000)
    wacc['Debt'] = int(self.getLongDebt()/1000)
    wacc['WACC'] = np.char.mod('%.2f%%', self.getWACC()*100)
    return wacc.T

  # function getMarketRiskPremium
  def getMarketRiskPremium(self):
    return self.marketRiskPremium

  # TRY TO GET THIS SHIT WITH ALPHA VANTAGE
  # function getRiskFreeRate
  def getRiskFreeRate(self):
    if self.riskFreeRate == 0.0:  
      ten_year = '^TNX'
      treasury_ticker = [ten_year]
      treasury_history = yf.download(treasury_ticker, period = '5d', progress=False)
      self.riskFreeRate = float(treasury_history['Adj Close'].iloc[-1])/100
    return self.riskFreeRate

  # function calculate Beta or get Input
  def calcBeta(self):
    beta = float(self.getOverview()['beta'])
    print("Current Beta is: " + str(beta))
    beta = float(input("Insert Beta value for DCF: "))
    self.beta = beta

  # function getBeta
  def getBeta(self):
    if self.beta == 0.0:
      self.calcBeta()
    return self.beta

  # function calculate Cost of Equity
  def getCostOfEquity(self):
    costOfEquity = self.getBeta() * self.getMarketRiskPremium() + self.getRiskFreeRate()
    return costOfEquity

  # function calcualte PreTax
  def getPreTaxCostOfDebt(self):
    interestExpense = self.get5YearDataIS('interestExpense')
    balanceSheet = self.getBalanceSheet()
    totalDebt = np.array(balanceSheet.iloc[:, ::-1].loc['shortLongTermDebtTotal']).astype('int64')  
    return float(interestExpense[4] / totalDebt[4]) + self.getRiskFreeRate()

  # function retrieve taxRate from yfinance
  def getTaxRate(self):
    stock = yf.Ticker(self.stock)
    taxRate = float(stock.get_financials().loc['TaxRateForCalcs'][0])
    return taxRate

  # function calculates After-Tax Cost of Debt
  def getAfterTaxCostOfDebt(self):
    return (self.getPreTaxCostOfDebt() * (1 - self.getTaxRate()))

  # retrieves current price
  def getCurrentPrice(self):
    currentPrice = self.ts.get_daily_adjusted(stock)[0].iloc[0].loc['4. close']
    return currentPrice

  # retrieves shares oustanding
  def getSharesOustanding(self):
    shares = int(self.getOverview()['sharesOutstanding'])
    return shares

  # calculates current equity value
  def getCurrentEquityValue(self):
    equity = self.getCurrentPrice() * self.getSharesOustanding()
    return equity

  # retrieves long-term debt
  def getLongDebt(self):
    balanceSheet = self.getBalanceSheet()
    longDebt = np.array(balanceSheet.iloc[:, ::-1].loc['longTermDebt']).astype('int64')
    return longDebt[4]

  # calculates WACC
  def getWACC(self):
    wacc = self.getAfterTaxCostOfDebt() * (self.getLongDebt() / (self.getLongDebt()+self.getCurrentEquityValue())) + self.getCostOfEquity() * (self.getCurrentEquityValue() / (self.getLongDebt()+self.getCurrentEquityValue()))
    return wacc

  # CALCULATIONS FOR GETTING NWC

  def get5YearBS(self, stri):
    bal_sheet = self.getBalanceSheet()
    return np.array(bal_sheet.iloc[:, ::-1].loc[stri])

  def get5YearAR(self):
    return self.get5YearBS('currentNetReceivables').astype('int64')

  def get5YearInv(self):
    arrInventory = self.get5YearBS('inventory')
    return arrInventory.astype('int64')

  def get5YearAP(self):
    return self.get5YearBS('currentAccountsPayable').astype('int64')

  #projects 10-year
  def projectDSO(self):
    fiveYearDSO = (self.get5YearAR() / self.get5YearDataIS('totalRevenue'))*365.0
    avgDSO = fiveYearDSO.mean()
    for _i in range(5,10):
      fiveYearDSO = np.append(fiveYearDSO, avgDSO)
    return fiveYearDSO

  def projectDIO(self):
    fiveYearDIO = (self.get5YearInv() / self.get5YearDataIS('costofGoodsAndServicesSold'))*365.0
    avgDIO = fiveYearDIO.mean()
    for _i in range(5,10):
      fiveYearDIO = np.append(fiveYearDIO, avgDIO)
    return fiveYearDIO

  def projectDPO(self):
    fiveYearDPO = (self.get5YearAP() / self.get5YearDataIS('costofGoodsAndServicesSold'))*365.0
    avgDPO = fiveYearDPO.mean()
    #fiveYearDPO = np.append(fiveYearDPO, np.full(5, avgDPO))
    for _i in range(0, 5):
      fiveYearDPO = np.append(fiveYearDPO, avgDPO)
    return fiveYearDPO

  def projectAR(self):
    tenYearDSO = self.projectDSO()
    revenue = self.projectRevenue()
    accountsRec = self.get5YearAR()
    for i in range (5,10):
      accountsRec = np.append(accountsRec, tenYearDSO[i]/365.0*revenue[i])
    return accountsRec

  def projectInv(self):
    tenYearDIO = self.projectDIO()
    cogs = self.projectCOGS()
    inventory = self.get5YearInv()
    for i in range(5,10):
      inventory = np.append(inventory, tenYearDIO[i]/365.0*cogs[i])
    return inventory

  def projectAP(self):
    tenYearDPO = self.projectDPO()
    accountsPayable = self.get5YearAP()
    cogs = self.projectCOGS()
    for i in range(5,10):
      accountsPayable = np.append(accountsPayable, tenYearDPO[i]/365*cogs[i])
    return accountsPayable

  def get5YearOtherAssets(self):
    otherAssets = self.get5YearBS('otherCurrentAssets')
    return otherAssets.astype('int64')

  def projectPercentOfSGA(self):
    otherCurrentAssets = self.get5YearOtherAssets()
    sga = self.get5YearDataIS('sellingGeneralAndAdministrative')
    percentOfSGA = otherCurrentAssets / sga
    avgPercentOfSGA = percentOfSGA.mean()
    for _i in range(0,5):
      percentOfSGA = np.append(percentOfSGA, avgPercentOfSGA)
    return percentOfSGA

  def projectOtherAssets(self):
    otherAssets = self.get5YearOtherAssets()
    percentOfSGA = self.projectPercentOfSGA()
    sga = self.projectSGA()
    for i in range(5,10):
      otherAssets = np.append(otherAssets, sga[i]*percentOfSGA[i])
    return otherAssets

  def get5YearAL(self):
    return self.get5YearBS('totalCurrentLiabilities').astype('int64')

  def projectPercentOfRevenueAL(self):
    fiveYearAL = (self.get5YearAL() / self.get5YearDataIS('totalRevenue'))
    avgFiveYearAL = fiveYearAL.mean()
    fiveYearAL = np.append(fiveYearAL, np.full(5, avgFiveYearAL)) 
    return fiveYearAL

  def projectAL(self):
    tenYearAL = self.get5YearAL()
    percentOfRevenueAL = self.projectPercentOfRevenueAL()
    revenue = self.projectRevenue()
    for i in range(5,10):
      tenYearAL = np.append(tenYearAL, percentOfRevenueAL[i]*revenue[i])
    return tenYearAL

  def projectTotalCurrentAssets(self):
    return self.projectAR() + self.projectInv() + self.projectOtherAssets()

  def projectTotalCurrentLiabilities(self):
    return self.projectAP() + self.projectAL()

  def projectNWC(self):
    return self.projectTotalCurrentAssets() - self.projectTotalCurrentLiabilities()

  def projectChangeNWC(self):
    change = np.array([0.0])
    nwc = self.projectNWC()
    for i in range(1,10):
      change = np.append(change, nwc[i] - nwc[i-1])
    return change

  # FUNCTIONS BELOW ARE FOR CASH FLOW ADJUSTMENTS
  def get5YearCapEx(self):
    return self.get5YearDataCF('capitalExpenditures').astype('int64')

  def projectPercentOfRevenueCapEx(self):
    capExPercentOfRev = self.get5YearCapEx() / self.get5YearDataIS('totalRevenue')
    avgPercent = capExPercentOfRev.mean()
    for i in range(5,10):
      capExPercentOfRev = np.append(capExPercentOfRev, avgPercent)
    return capExPercentOfRev

  def projectCapEx(self):
    capEx = self.get5YearCapEx()
    capExPercentOfRev = self.projectPercentOfRevenueCapEx()
    totalRevenue = self.projectRevenue()
    for i in range(5, 10):
      capEx = np.append(capEx, capExPercentOfRev[i]*totalRevenue[i])
    return capEx

  def get5YearDepAmort(self):
    return self.get5YearDataCF('depreciationDepletionAndAmortization').astype('int64')

  def projectPercentOfCapEx(self):
    fiveYearCapEx = self.get5YearCapEx()
    percentOfCapEx = (1.0*self.get5YearDepAmort()) / fiveYearCapEx
    multiplier = (1.0 / percentOfCapEx[4])**(1/5)
    for i in range(4,9):
      percentOfCapEx = np.append(percentOfCapEx, percentOfCapEx[i] * multiplier)
    return percentOfCapEx

  def projectDepAmort(self):
    depAmort = self.get5YearDepAmort()
    percentOfCapEx = self.projectPercentOfCapEx()
    capEx = self.projectCapEx()
    for i in range(5,10):
      depAmort = np.append(depAmort, percentOfCapEx[i]*capEx[i])
    return depAmort

  def projectTaxRate(self):
    taxRate = self.getTaxRate()
    arr = np.full(10, taxRate)
    return arr

  def projectEBITMargin(self):
    revenue = self.projectRevenue()
    ebit = self.projectEBIT()
    ebitMargin = ebit / revenue
    return ebitMargin

  def projectNOPAT(self):
    ebit = self.projectEBIT()
    taxRate = self.projectTaxRate()
    nopat = ebit * (1 - taxRate)
    return nopat

  def projectFCFF(self):
    return self.projectNOPAT() + self.projectDepAmort() - self.projectCapEx() - self.projectChangeNWC()

  def projectFCFFPercentGrowth(self):
    fcff = self.projectFCFF()
    fcffPercentGrowth = np.array([0.0])
    for i in range(1,10):
      fcffPercentGrowth = np.append(fcffPercentGrowth, (fcff[i] - fcff[i-1])/fcff[i-1])
    return fcffPercentGrowth

  def projectDiscountPeriod(self):
    arr = np.zeros(5)
    arr = np.append(arr, [0.25, 1.25, 2.25, 3.25, 4.25])
    return arr

  def projectDiscountFactor(self):
    wacc = self.getWACC()
    discountPeriod = self.projectDiscountPeriod()
    discountFactor = 1 / ((1 + wacc)**discountPeriod)
    return discountFactor

  def projectPVCashFlow(self):
    fcff = self.projectFCFF()
    discountFactor = self.projectDiscountFactor()
    pv = fcff * discountFactor
    return pv

  # FUNCTIONS BELOW FOR GORDON GROWTH METHOD
  def getTerminalCF(self):
    terminalCF = self.projectFCFF()[-1]
    return terminalCF

  def getTerminalGrowthRate(self):
    return self.terminalGrowthRate

  def getTerminalValue(self):
    terminalCF = self.getTerminalCF()
    wacc = self.getWACC()
    terminalGrowthRate = self.getTerminalGrowthRate()
    terminalValue = terminalCF * (1 + terminalGrowthRate) / (wacc - terminalGrowthRate)
    return terminalValue

  def getPVOfTerminalValue(self):
    terminalValue = self.getTerminalValue()
    wacc = self.getWACC()
    discountFactor = self.projectDiscountFactor()
    presentValue = terminalValue / (1 + wacc)**discountFactor[-1]
    return presentValue

  def getEnterpriseValue(self):
    pvCashFlow = np.sum(self.projectPVCashFlow()[5:])
    pvTerminalValue = self.getPVOfTerminalValue()
    return pvCashFlow + pvTerminalValue

  def getCash(self):
    cash = self.get5YearBS('cashAndCashEquivalentsAtCarryingValue')[-1]
    return int(cash)

  def getEquityValueGG(self):
    enterpriseValue = self.getEnterpriseValue()
    debt = self.getLongDebt()
    cash = self.getCash()
    equityValue = enterpriseValue - debt + cash
    return equityValue

  def getPricePerShareGG(self):
    equityValue = self.getEquityValueGG()
    shares = self.getSharesOustanding()
    return 1.0*equityValue/shares

  # FUNCTIONS BELOW ARE FOR MULTIPLES METHOD
  def getTerminalEBITDA(self):
    ebit = self.projectEBIT()
    depAmort = self.projectDepAmort()
    terminalEBITDA = ebit[-1] + depAmort[-1]
    return terminalEBITDA

  def calcEVEBITDAMultiple(self):
    evebitda = float(self.getOverview()['enterpriseToEbitda'])
    print("Current EV/EBITDA Multiple is: " + str(evebitda))
    evebitda = float(input("Insert EV/EBITDA value for DCF: "))
    self.evebitda = evebitda

  def getEVEBITDAMultiple(self):
    if self.evebitda == 0.0:
      self.calcEVEBITDAMultiple()
    return self.evebitda

  def getTerminalValueMM(self):
    terminalEBITDA = self.getTerminalEBITDA()
    multiple = self.getEVEBITDAMultiple()
    terminalValue = terminalEBITDA * multiple
    return terminalValue

  def getPVOfTerminalValueMM(self):
    terminalValue = self.getTerminalValueMM()
    wacc = self.getWACC()
    discountFactor = self.projectDiscountFactor()
    presentValue = terminalValue / (1 + wacc)**discountFactor[-1]
    return presentValue

  def getEnterpriseValueMM(self):
    pvCF = np.sum(self.projectPVCashFlow()[5:])
    pvTerminalValue = self.getPVOfTerminalValueMM()
    return pvCF + pvTerminalValue

  def getEquityValueMM(self):
    enterpiseValuse = self.getEnterpriseValueMM()
    debt = self.getLongDebt()
    cash = self.getCash()
    equityValue = enterpiseValuse - debt + cash
    return equityValue

  def getPricePerShareMM(self):
    return 1.0*self.getEquityValueMM()/self.getSharesOustanding()

stock = input("Insert a stock Ticker: ")
dcf = DCF(stock, '7L5FSOHGNSR3M4YL')
#2XQGIZJ4GO9E8A60
#K0FGF8K2SAV8NSHT
#paid 01IN1MCJXO7P7AVK
rev = dcf.createRevenueDF()
cfa = dcf.createCFA()
nwc = dcf.createNWC()
fcf = dcf.createFCF()
wacc = dcf.createWACC()
gord = dcf.createGordonGrowth()
mult = dcf.createMultiples()
print(rev)
print()
print(cfa)
print()
print(nwc)
print()
print(fcf)
print()
print(wacc)
print()
print(gord)
print()
print(mult)


# Note: find a way to BOLD certain rows in the DataFrames
# Note: add way to add user input for EV/EBITDA multiple value
# NOTE: IMPORTANT CHECK TO SEE IF PERCENT FUNCTIONS GIVE A FLOAT (PERCENT) SINCE WE CHANGED ALL ARRAYS TO BE OF TYPE INT
