
from random import random
import numpy as np
import plotly.express as px
import pymysql
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Dimension of truck
space_limit_truck={'length':2,'width':1,'height':1}

# Store product attributes
class Product: 
    def __init__(self, name, length, width, height, weight, price):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.price = price
# Represent an individual in a population and is a potential solution
class Individual: 
    def __init__(self,lengths, widths, heights, spaces, prices, weights, space_limit, weight_limit, generation=0):
        self.lengths = lengths
        self.widths = widths
        self.heights = heights
        self.spaces = spaces
        self.prices = prices
        self.weights = weights
        self.space_limit = space_limit
        self.weight_limit = weight_limit
        self.score_evaluation = 0
        self.used_space = 0
        self.used_weight = 0
        self.generation = generation
        self.chromosome = []

        for i in range(len(spaces)):
            if random() < 0.5:
                self.chromosome.append('0') # Product is not selected
            else:
                self.chromosome.append('1') # Product is selected
    
    # Decimal count function of product dimension
    def count_decimal_places(self):
        result = 1
        for index, product in enumerate(self.spaces):

            formatted_height = f"{float(self.heights[index])}"
            formatted_width = f"{float(self.widths[index])}"
            formatted_length = f"{float(self.lengths[index])}"

            decimal_part1 = formatted_height.split('.')[1]
            decimal_part2 = formatted_width.split('.')[1]
            decimal_part3 = formatted_length.split('.')[1]

            decimal_part1 = decimal_part1.rstrip('0')
            decimal_part2 = decimal_part2.rstrip('0')
            decimal_part3 = decimal_part3.rstrip('0')
            result = max(len(decimal_part1), len(decimal_part2), len(decimal_part3), result)
        return result
    
    # The can_fit function determines whether an item can be placed into space at a specific location without exceeding a size limit.
    def can_fit(self, index_item, spaces, position):
        x_pos, y_pos, z_pos = position
        decimal = self.count_decimal_places()
        decimal = 10**decimal 
        if (x_pos + int(self.lengths[index_item]*decimal) > int(space_limit_truck['length']*decimal) or \
           y_pos + int(self.widths[index_item]*decimal) > int(space_limit_truck['width']*decimal) or \
           z_pos + int(self.heights[index_item]*decimal) > int(space_limit_truck['height']*decimal)):
           return False
        for z in range(0,int(self.heights[index_item]*decimal)):
            for x in range(0,int(self.lengths[index_item]*decimal)):
                for y in range(0,int(self.widths[index_item]*decimal)):
                    if spaces[x_pos + x][y_pos + y][z_pos + z] == 1:
                        return False
        return True

    # The replace item function marks spaces as occupied by items.
    def place_item(self, index_item, spaces, position):
        x_pos, y_pos, z_pos = position
        decimal = self.count_decimal_places()
        decimal = 10**decimal 
        for z in range(int(self.heights[index_item]*decimal)):
            for x in range(int(self.lengths[index_item]*decimal)):
                for y in range(int(self.widths[index_item]*decimal)):
                        spaces[x_pos + x][y_pos + y][z_pos + z] = 1

    # The check space function checks whether all items in the chromosome can be placed in the bin space without violations.
    def check_space(self):
        decimal = self.count_decimal_places()
        decimal = 10**decimal
        space_3d = np.zeros((space_limit_truck['length']*decimal, space_limit_truck['width']*decimal, space_limit_truck['height']*decimal), dtype=int)
        total_sum = sum(1 for i in self.chromosome if i == '1')
        count_item = 0
        for i, included in enumerate(self.chromosome):
          if included == '1':
              placed = False
              for z in range(int(space_limit_truck['height'] * decimal)):
                  for x in range(int(space_limit_truck['length'] * decimal)):
                      for y in range(int(space_limit_truck['width'] * decimal)):
                          position = (x, y, z)
                          if self.can_fit(i, space_3d, position):
                              count_item += 1
                              self.place_item(i, space_3d, position)
                              placed = True
                              break
                      if placed:
                          break
                  if placed:
                      break

        return count_item == total_sum
    # Fitness function calculates fitness of each individual in population
    def fitness(self):
        score = 0
        sum_spaces = 0
        sum_weights = 0
        for i in range(len(self.chromosome)):
            if self.chromosome[i] == '1':
                score += self.prices[i]
                sum_weights += self.weights[i]
                sum_spaces += self.spaces[i]
        if sum_spaces > self.space_limit or sum_weights > self.weight_limit or self.check_space() == False:
            score = 0 #Penalty if not suitable
        self.score_evaluation = score
        self.used_space = sum_spaces
        self.used_weight = sum_weights
        return score
    
    # Crossover function is used to cross two sets of chromosome from father and mother to create an offspring 
    def crossover(self, other_individual):
        cutoff = round(random() * len(self.chromosome))
        child1 = other_individual.chromosome[0:cutoff]+self.chromosome[cutoff::]
        child2 = self.chromosome[0:cutoff] + other_individual.chromosome[cutoff::]
        children = [Individual(self.spaces,self.lengths,self.widths,self.heights, self.prices, self.weights, self.space_limit, self.weight_limit, self.generation + 1),
                    Individual(self.spaces,self.lengths,self.widths,self.heights, self.prices, self.weights, self.space_limit, self.weight_limit, self.generation + 1)]
        children[0].chromosome = child1
        children[1].chromosome = child2
        return children
    # Mutation function creates diversity in a population by mutating each gene of each individual
    def mutation(self, rate):
      for i in range(len(self.chromosome)):
            if random() < rate:
                if self.chromosome[i] == '1':
                    self.chromosome[i] = '0'
                else:
                    self.chromosome[i] = '1'
      return self
    
#Using Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.generation = 0
        self.best_solution = None
        self.population = []
        self.list_of_solution = []
    def initializePopulation(self,lengths,widths,heights,spaces, prices, weights, space_limit, weight_limit):
        for i in range(self.population_size):
            self.population.append(Individual(lengths,widths,heights,spaces,prices,weights,space_limit,weight_limit))
        self.best_solution = self.population[0]

    def orderPopulation(self):
        self.population = sorted(self.population, key=lambda population: population.score_evaluation, reverse = True)

    def best_individual(self, individual):
      if individual.score_evaluation > self.best_solution.score_evaluation:
        self.best_solution = individual

    def sum_evaluations(self):
      sum_temp = 0
      for individual in self.population:
        sum_temp+= individual.score_evaluation
      return sum_temp

    def select_parent(self, sum_evaluations):
      parent = -1
      random_value = random() * sum_evaluations
      sum_temp = 0
      i = 0
      while i < len(self.population) and sum_temp < random_value:
        sum_temp += self.population[i].score_evaluation
        parent += 1
        i += 1
      return parent
    def visualize_generation(self):
      best = self.population[0]
      print('Generation: {}, '
            'Total price: {:.3f}, '
            'Space: {:.3f}, '
            'Weight: {:.3f}, '
            'Chromosome: {}'.format(
                self.population[0].generation,
                best.score_evaluation,
                best.used_space,
                best.used_weight,
                best.chromosome))

    def solve(self, mutation_probability, number_of_generation,lengths,widths,heights, spaces, weights, prices, limit_space, limit_weight):
      self.initializePopulation(lengths,widths,heights,spaces, weights, prices, limit_space, limit_weight)

      for individual in self.population:
        individual.fitness()
      self.orderPopulation()
      self.best_solution = self.population[0]

      self.list_of_solution.append(self.best_solution.score_evaluation)

      self.visualize_generation()
      for generation in range(number_of_generation):
        sum_temp = self.sum_evaluations()
        new_population = []
        for new_individual in range(0, self.population_size, 2):
          parent1 = self.select_parent(sum_temp)
          parent2 = self.select_parent(sum_temp)
          children = self.population[parent1].crossover(self.population[parent2])
          new_population.append(children[0].mutation(mutation_probability))
          new_population.append(children[1].mutation(mutation_probability))
        self.population = list(new_population)

        for individual in self.population:
          individual.fitness()
        self.visualize_generation()
        best = self.population[0]
        self.list_of_solution.append(best.score_evaluation)
        self.best_individual(best)

      print(' **** Best solution - Generation: {}, '
            'Total price: {:.3f}, '
            'Space: {:.3f}, '
            'Weight: {:.3f}, '
            'Chromosome: {}'.format(
                self.best_solution.generation,
                self.best_solution.score_evaluation,
                self.best_solution.used_space,
                self.best_solution.used_weight,
                self.best_solution.chromosome))


      return self.best_solution.chromosome

products_list = []
# Connect to MySql
connection = pymysql.connect(host='localhost', user='root', passwd='01656107662aA&', db='products')
cursor = connection.cursor()
cursor.execute('select product_name, product_length, product_width, product_height, weight, price, quantity from products;')
for product in cursor:
    for i in range(product[6]):
      products_list.append(Product(product[0],product[1],product[2],product[3],product[4],product[5]))
cursor.close()
connection.close()

width = []
height = []
length = []
weight = []
price = []
name = []
spaces = []
for item in products_list:
  width.append(item.width)
  height.append(item.height)
  length.append(item.length)
  price.append(item.price)
  name.append(item.name)
  weight.append(item.weight)
  spaces.append(item.width*item.height*item.length)

limit_weight = 150
limit_space = space_limit_truck['height']*space_limit_truck['length']*space_limit_truck['width'] #Gioi han khong gian
population_size = 20
mutation_probability = 0.01
number_of_generation = 100
products = []

GA = GeneticAlgorithm(population_size)
result = GA.solve(mutation_probability, number_of_generation,length,width,height, spaces, price, weight, limit_space, limit_weight)
print()
for i in range(len(products_list)):
  if result[i] == '1':
    print('Name: ', products_list[i].name, ' - Price: ', products_list[i].price)
    products.append(products_list[i])
# Modaling data to see which option is best
figure = px.line(x = range(0,101), y = GA.list_of_solution, title = 'Genetic algorithm results')
figure.show()

lengths = []
widths = []
heights = []
weights = []
prices = []
spaces = []
for i in products:
  lengths.append(i.length)
  widths.append(i.width)
  heights.append(i.height)
  weights.append(i.weight)
  prices.append(i.price)
  spaces.append(i.length*i.width*i.height)

weight_limit = 150 
best_individual = Individual(lengths, widths, heights, spaces, prices, weights, space_limit_truck, weight_limit)
decinimal = best_individual.count_decimal_places()
decinimal = 10**decinimal
Products = []
for i in products:
  Products.append({'height':i.height,'width':i.width,'length':i.length})

# 3D modeling of products loaded into the vehicle's container

def can_fit(item, box_dimensions, space, position):
    x_pos, y_pos, z_pos = position
    if x_pos + item['length'] > box_dimensions['length'] or \
       y_pos + item['width'] > box_dimensions['width'] or \
       z_pos + item['height'] > box_dimensions['height']:
        return False

    for z in range(item['height']):
        for x in range(item['length']):
            for y in range(item['width']):
                if space[x_pos + x][y_pos + y][z_pos + z] == 1:  # Nếu ô đã bị chiếm
                    return False
    return True
def place_item(item, space, position):
    x_pos, y_pos, z_pos = position
    for z in range(item['height']):
        for x in range(item['length']):
            for y in range(item['width']):
                space[x_pos + x][y_pos + y][z_pos + z] = 1
# Redraw products after putting the items in.
def draw_box(space, box_dimensions, items_in_box):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw container
    ax.scatter(0, 0, 0, color='gray', s=100, alpha=0.5, label='Box')

    # Draw the placed items
    colors = plt.cm.viridis(np.linspace(0, 1, len(items_in_box)))
    for index, item in enumerate(items_in_box):
        x_pos = item['position'][0]
        y_pos = item['position'][1]
        z_pos = item['position'][2]

        # Create rectangles for each item
        ax.bar3d(x_pos, y_pos, z_pos, item['length'], item['width'], item['height'], color=colors[index], alpha=0.6)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Packing Visualization')
    ax.set_xlim(0, box_dimensions['length'])
    ax.set_ylim(0, box_dimensions['width'])
    ax.set_zlim(0, box_dimensions['height'])
    plt.legend()
    plt.show()
# function to put items in the container
def draw_box_animation(space, box_dimensions, items_in_box):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits
    ax.set_xlim(0, box_dimensions['length'])
    ax.set_ylim(0, box_dimensions['width'])
    ax.set_zlim(0, box_dimensions['height'])

    # Draw items in the container
    colors = plt.cm.viridis(np.linspace(0, 1, len(items_in_box)))
    for index, item in enumerate(items_in_box):
        x_pos, y_pos, z_pos = item['position']
        ax.bar3d(x_pos, y_pos, z_pos, item['length'], item['width'], item['height'], color=colors[index], alpha=0.6)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Packing Visualization')

    # Function to update rotation angle for each frame
    def update(angle):
        ax.view_init(elev=20, azim=angle)

    # Create animation to rotate from 0 to 180 degrees
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 180, num=36), interval=100)
    plt.show()

# Packing items function
def pack_items(items):
    box_dimensions = {'height': int(1*decinimal), 'length': int(2*decinimal), 'width': int(1*decinimal)}
    space = np.zeros((box_dimensions['length'], box_dimensions['width'], box_dimensions['height']), dtype=int)
    items_in_box = []
    i=0
    for item in items:
        item = {'height': int(item['height']*decinimal),
                'length': int(item['length']*decinimal),
                'width': int(item['width']*decinimal)}
        placed = False
        for z in range(box_dimensions['height']):
            for x in range(box_dimensions['length']):
                for y in range(box_dimensions['width']):
                    position = (x, y, z)
                    if can_fit(item, box_dimensions, space, position):
                        items_in_box.append({**item, 'position': position})
                        place_item(item, space, position)
                        placed = True
                        i+=1
                        print(i)
                        # Draw after adding each item
                        draw_box(space, box_dimensions, items_in_box)
                        break 
                if placed:
                    break 
            if placed:
                break 


    draw_box_animation(space, box_dimensions, items_in_box)
    return items_in_box


packed_item = pack_items(Products)





