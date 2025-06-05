# class and object

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hi, I'm {self.name} and I'm {self.age} years old.")

p1 = Person("diya",21)
p1.introduce()


#Encapsulation

class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance


account = BankAccount("Bob", 1000)
account.deposit(500)
print(account.get_balance()) 


# abstraction

from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Cat(Animal):
    def make_sound(self):
        print("Meow")


cat = Cat()
cat.make_sound()



# inheritance

class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def drive(self):
        print(f"{self.brand} is moving")

class Car(Vehicle): 
    def honk(self):
        print(f"{self.brand} says honk honk!")

my_car = Car("Toyota")
my_car.drive()
my_car.honk()


# polymorphism

class Bird:
    def sound(self):
        print("Chirp")

class Duck(Bird):
    def sound(self):
        print("Quack")

class Crow(Bird):
    def sound(self):
        print("Caw")


def make_bird_sound(bird):
    bird.sound()

duck = Duck()
crow = Crow()

make_bird_sound(duck)
make_bird_sound(crow)

