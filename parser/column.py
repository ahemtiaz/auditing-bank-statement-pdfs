import dateparser
import re
from typing import TypeVar, Generic, Type
from typing import get_origin, get_args

def get_type_name(cls):
    origin = get_origin(cls)
    args = get_args(cls)
    if origin is None:
        return cls.__name__
    elif len(args) == 0:
        return origin.__name__
    else:
        args = [get_type_name(arg) for arg in args]
        return f"{origin.__name__}[{', '.join(args)}]"


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        s for c in cls.__subclasses__() for s in get_all_subclasses(c)
    )

class Data:
    def __init__(self):
        self.value = None

    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__ method")
    

    @staticmethod
    def type(header: str):
        subclasses = get_all_subclasses(Data)
        for subclass in subclasses:
            factory = subclass.match_header(header)
            if factory is not None:
                return factory
        return None

    @classmethod
    def match_header(cls, header: str):
        header = header.replace("/", " ")
        header = header.replace(".", " ")
        header = header.strip()
        if hasattr(cls, 'header_pattern'):
            if re.fullmatch(cls.header_pattern, header, re.IGNORECASE) is not None:
                return DataFactory(cls)
        return None
    
    @classmethod
    def match_data(cls, data: str):
        if hasattr(cls, 'data_pattern'):
            match = re.search(cls.data_pattern, data) 
            if match is not None:
                return match.group(0)
        return None
    
    def get_columns(self):
        return [self]
    
class DataFactory:
    def __init__(self, cls, M=None, N=None):
        self.cls = cls
        self.M = M
        self.N = N

    def create(self, value: str):
        if get_origin(self.cls) is Union or get_origin(self.cls) is Composite:
            if self.M is None or self.N is None:
                raise ValueError("M and N must be provided for Union or Composite")
            return self.cls(self.M, self.N, value)
        else:
            return self.cls(value)

class CompositeBase(Data):
    @classmethod
    def match_header(cls, header: str):
        header = header.replace(" AND ", " and ")
        if ' and ' not in header:
            return None
        split = header.split(' and ')
        header1 = split[0].strip()
        header2 = split[1].strip()

        cls1 = Data.type(header1).cls
        cls2 = Data.type(header2).cls

        if cls1 is not None and cls2 is not None:
            return DataFactory(Composite[cls1, cls2], cls1, cls2)
        
        return None

M = TypeVar("M", bound=Data)
N = TypeVar("N", bound=Data)
class Composite(CompositeBase, Generic[M, N]):
    def __init__(self, Mcls: Type[M], Ncls: Type[N], value: str):
        self.M = Mcls
        self.N = Ncls
        
        value1, value2 = self.match_data(self.M, self.N, value)

        if value1 is None or value2 is None:
            raise ValueError(f"Invalid data format for {self.M.name} and {self.N.name}: {value}")
        
        self.data = self.M(value1), self.N(value2)
    
    @classmethod
    def match_data(cls, Mcls: Type[M], Ncls: Type[N], value: str):
        value1 = Mcls.match_data(value)
        if value1 is None:
            return None
        
        value2 = value.replace(value1, "", 1)
        value2 = Ncls.match_data(value2)
        if value2 is None:
            return None
        
        return value1, value2
    
    def __str__(self):
        return f"{self.data[0]} and {self.data[1]}"
    
    def get_columns(self):
        data1, data2 = self.data
        return [data for data in [data1, data2] if data is not None]
    

class UnionBase(Data):
    @classmethod
    def match_header(cls, header: str):
        if '/' not in header:
            return None
        split = header.split('/')
        split = [s.strip() for s in split]

        clss = []
        i = 0
        while i < len(split):
            data = split[i]
            cls = Data.type(data)
            while i < len(split) - 1 and cls is None:
                data = data + ' ' + split[i + 1]
                cls = Data.type(data)
                i += 1
            clss.append(cls)
            i += 1
        if len(clss) == 2:
            cls1, cls2 = clss[0].cls, clss[1].cls

            return DataFactory(Union[cls1, cls2], cls1, cls2)
        return None
    
class Union(UnionBase, Generic[M, N]):
    def __init__(self, Mcls: Type[M], Ncls: Type[N], value: str):
        self.M = Mcls
        self.N = Ncls

        self.data: Union[Data, Union[Data, Data]] 
        match1, match2 = self.match_data(self.M, self.N, value)

        if match1 is None and match2 is None:
            raise ValueError(f"Invalid data format for {self.M.name} or {self.N.name}: {value}")
        
        if match1 is None:
            self.data = None, self.N(match2)
        elif match2 is None:
            self.data = self.M(match1), None
        else:
            self.data = self.M(match1), self.N(match2)

    @classmethod
    def match_data(cls, Mcls: Type[M], Ncls: Type[N], value: str):
        if value.count('/') == 2:
            split = value.split('/')
            split = [s.strip() for s in split]
            value1 = split[0]
            value2 = split[1]
            value1 = Mcls.match_data(value1)
            value2 = Ncls.match_data(value2)
            return value1, value2
        elif '/' not in value:
            match1 = Mcls.match_data(value)
            match2 = Ncls.match_data(value)
            return match1, match2

    
    def __str__(self):
        return f"{self.data[0]} or {self.data[1]}"
    
    def get_columns(self):
        data1, data2 = self.data
        return [data for data in [data1, data2] if data is not None]


class Time(Data):
    name = "Time"
    header_pattern = r"(tm|trn|trans|book|posting)?\s*time"
    def __init__(self, value: str):
        value = dateparser.parse(value)
        if not value:
            raise ValueError(f"Invalid time format: {value}")
        
        self.hour = value.hour
        self.minute = value.minute
        
    def __str__(self):
        return f"{self.hour:02d}:{self.minute:02d}"

class String(Data):
    data_pattern = r".*"

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value

class UUID(String):
    data_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

class Boolean(Data):
    data_pattern = r"(t|T)rue|(f|F)alse|(y|Y)es|(n|N)o|1|0"

    def __init__(self, value: str):
        if value.lower() in ["true", "yes", "1"]:
            self.value = True
        elif value.lower() in ["false", "no", "0"]:
            self.value = False
        else:
            raise ValueError(f"Invalid boolean value: {value}")

    def __str__(self):
        return str(self.value)
    
class MultilineString(String):
    data_pattern = r"(.|\n)*"

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value
    
class Amount(Data):
    data_pattern = r"(\d{1,3}(,\d{2,3})*(\.\d{2})?)|(\d+(\.\d{2})?)"

    def __init__(self, value: str):
        self.value = float(value.replace(",", ""))
    def __str__(self):
        return f"{self.value:,.2f}"

class NumericCode(String):
    data_pattern = r"\d+"

class AlphaNumericCode(String):
    data_pattern = r"[a-zA-Z0-9]+"

class Integer(Data):
    data_pattern = r"\d+"

    def __init__(self, value: str):
        try:
            self.value = int(value)
        except ValueError:
            raise ValueError(f"Invalid integer number: {value}")
    def __str__(self):
        return str(self.value)
    

class Date(Data):
    name = "Date"
    header_pattern = r"(trn|trans|book|post|posting)?\s*date"
    data_pattern = r".*" # too generic, but we will parse it later
    def __init__(self, value: str):
        value = self.match_data(value)
        value = dateparser.parse(value, date_formats=[
                                        "%d-%m-%Y", 
                                        "%d/%m/%Y",
                                        "%d %m %Y",
                                        "%d-%m-%y",
                                        "%d/%m/%y",
                                        "%d %m %y",
                                        "%d-%b-%Y",
                                        "%d/%b/%Y",
                                        "%d %b %Y",
                                        "%d %b %y",
                                        "%d-%b-%y",
                                        "%d/%b/%y",
                                        "%Y-%m-%d",
                                        "%Y/%m/%d",
                                        "%Y %m %d",
                                        "%Y-%m-%d %H:%M:%S",
                                        "%Y/%m/%d %H:%M:%S",
                                        "%Y %m %d %H:%M:%S",               
                                 ])
        if not value:
            raise ValueError(f"Invalid date format: {value}")
        
        self.day = value.day
        self.month = value.month
        self.year = value.year
        
    def __str__(self):
        return f"{self.day:02d}/{self.month:02d}/{self.year}"
        

class ValueDate(Date):
    name = "Value Date"
    header_pattern = r"(v|value)\s*date"
    
class Description(MultilineString):
    name = "Description"
    header_pattern = r"(trans|trn|transaction)?\s*(description|desc|detail|narration|narrative|particular|remark)s?|transaction"

class TransactionType(String):
    name = "Transaction Type"
    header_pattern = r"(trn|trans|transaction)\s*(type|mod)"

    
class Credit(Amount):
    name = "Credit"
    header_pattern = r"(cr|cr?edit|deposit)s?\s*(amt|amount)?\s*(\(cr\)|\(credit\))?"

class Debit(Amount):
    name = "Debit"
    header_pattern = r"(dr|debit|withdraw(al)?)s?\s*(amt|amount)?\s*(\(dr\)|\(debit\))?"

class Balance(Amount):
    name = "Balance"
    header_pattern = r"(curr|current)?\s*(bal|balanced?)\s*(amt|amount)?"

class Reference(AlphaNumericCode):
    name = "Reference"
    header_pattern = r"(trn)?\s*(ref|ref#?|references?)\s*(#|no)?"

class Cheque(NumericCode):
    name = "Cheque"
    data_pattern = r"\d{6,}"
    header_pattern = r"(chq|cheque|check)\s*(#|no)?"


class Serial(Integer):
    name = "Serial"
    header_pattern = r"(sl|si)#?"
    
class BranchCode(NumericCode):
    name = "Branch Code"
    header_pattern = r"(trn|transaction)?\s*(branch|brn?)\s*(code|id|no)?"

class BranchName(String):
    name = "Branch Name"
    header_pattern = r"(originating)\s*(branch|brn?)\s*(name|title)?"

class InstrumentNumber(AlphaNumericCode):
    name = "Instrument Number"
    header_pattern = r"(trn|trans|transaction)?\s*(instrument|instr?)\s*(#|no|number)?"

class TransactionCode(NumericCode):
    name = "Transaction Code"
    header_pattern = r"(trn|trans|transaction)?\s*(code)"

class BatchNumber(NumericCode):
    name = "Batch Number"
    header_pattern = r"(trn|trans|transaction)?\s*(batch)\s*(no|number)?"

class TracerNumber(NumericCode):
    name = "Tracer Number"
    header_pattern = r"(trn|trans|transaction)?\s*(tracer|trace)\s*(no|number)?"

class BankName(String):
    name = "Bank Name"
    header_pattern = r"(bank|banking)_?\s*(name|title)?"

class AccountNumber(NumericCode):
    name = "Account Number"
    data_pattern = r"\d{6,}"
    header_pattern = r"(stmt|statement)?_?\s*(acc|account)_?\s*(no|number)?"

class FileId(UUID):
    name = "File ID"
    header_pattern = r"(file)_?\s*(id|identifier|#)?"

class PassedValidation(Boolean):
    name = "Passed Validation"
    header_pattern = r"(passed)?_?\s*(validation|check)?"

class Page(Integer):
    name = "Page"
    header_pattern = r"(page|pg)_?\s*(#|no|index)?"


INCLUDE_COLUMNS = [
    FileId,
    Page,
    Serial,
    Time,
    Date,
    ValueDate,
    Reference,
    Cheque,
    BatchNumber,
    TracerNumber,
    InstrumentNumber,
    TransactionCode,
    TransactionType,
    Description,
    BankName,
    BranchCode,
    BranchName,
    AccountNumber,
    Debit,
    Credit,
    PassedValidation,
]
