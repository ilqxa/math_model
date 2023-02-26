from pydantic import Field

from base.operations import Operation, Addition, Multiplication, Exponentiation
from base.values import Value, Constant, Variable


class Derivative(Operation):
    """Операция взятия производной от заданной функции вида f^order(func), где
    func - исходная функция
    order - порядок производной
    """
    func: Operation | Value = Field(description='исходная функция')
    order: int = Field(description='порядок производной')

    def __str__(self) -> str:
        return self.func.__str__() + "'"
    
    @staticmethod
    def find_base_solution(func: Operation | Value) -> Operation | Value:
        """Поиск решения по таблице производных простейших элементарных функций

        Args:
            base (Operation | Value): Исходная функция

        Raises:
            Exception: Неизвестная исходная функция

        Returns:
            Operation | Value: Первая производная исходной функции
        """
        if isinstance(func, Constant):
            # C' = 0
            return Constant(meaning=0., sign='0')
        elif isinstance(func, Variable):
            # x' = 1
            return Constant(meaning=1., sign='1')
        elif isinstance(func, Exponentiation):
            # (x^n)' = n*x^(n-1)
            return Multiplication(multiplers=[
                func.power,
                Exponentiation(
                    base=func.base,
                    power=Addition(terms=[func.power, Constant(meaning=-1, sign='-1')])
                ),
            ])
        raise Exception('Неизвестная функция, невозможно определить производную')
    
    @property
    def solution(self) -> Operation | Value:
        """Производная функции
        """
        der = self.func
        for i in range(1, self.order + 1):
            der = Derivative.find_base_solution(der)
        return der