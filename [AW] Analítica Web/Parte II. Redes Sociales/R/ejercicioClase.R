
# Leer el conjunto de datos del club de karate

nodes <- read.csv("nodos.csv", header = T)
links <- read.csv("enlaces.csv", header = T)

#Crear una red a partir de dichos datos (igraph)
net <- graph.data.frame(d=links,vertices=nodes,directed= F)

#Comprobar que el objeto net es de tipo igraph
class(net)

#Ver un resumen del tipo de dato
summary(net)

#Ver las primeras líneas de nodos y enlaces
head(nodes)
head(links)

#Acceder a las propiedades de los nodos y enlaces dentro del objeto net
V(net)
E(net)

#Acceder al campo label en los nodos y el campo weight en las aristas
table(V(net)$label)
table(E(net)$Weight)

plot(net)

#Utilizar las características de plot en igraph para optimizar el grafo
#vertex.color: color del nodo
#vertex.frame.color: color del borde del nodo
#vertex.label: etiquetas de los nodos
#vertex.label.cex: tamaño de letra de las etiquetas de los nodos
#...
#?igraph.plotting
plot(net, edge.arrow.size=.2, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5)
    
#Solamente mostrar las etiquetas sin nodos
plot(net, edge.arrow.size=.2, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5,vertex.shape="none")


#Utilizar layout para mostrar los nodos

#El layout aleatorio
plot(net, edge.arrow.size=.2, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5, vertex.size=5, layout=layout_randomly)

#Fruchterman-Reingold es uno de los layout basados en fuerza más utilizados
plot(net, edge.arrow.size=.2, edge.arrow.width=0.6, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5, vertex.size=5, layout=layout.fruchterman.reingold)



#Medidas generales de red: diámetro, densidad
diameter(net)
edge_density(net)

#Medidas generales de nodo: grado, intermediación

#Grado (se incluye como propiedad del nodo)
nodes$degree.total <- degree(net, v=V(net), mode="all")
nodes$degree.in <- degree(net,v=V(net), mode="in")
nodes$degree.out <- degree(net, v=V(net), mode = "out")

#Mostrar los 10 nodos con mejor de grado total
head(nodes[order(nodes$degree.total, decreasing= TRUE),], n=10L)

#Incorpor la información dentro del grafo
V(net)$outdegree <- degree(net, mode="out")
V(net)$indegree <- degree(net, mode="in")
V(net)$degree <- degree(net, mode="all")

#Intermediación
nodes$betweenness <- betweenness(net, v=V(net), directed=F,weights=NA)
V(net)$betweenness <- betweenness(net, directed=F,weights=NA)

#Dibujar el grafo con el tamaño del nodo en función del grado total
plot(net, edge.arrow.size=.2, edge.arrow.width=0.6, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5, vertex.size=V(net)$degree, layout=layout.fruchterman.reingold)

#Dibujar el grafo con el grosor de la arista en función del peso
plot(net, edge.arrow.size=.2, edge.arrow.width=0.6, edge.curved=0,
     vertex.color="gray", vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5, vertex.size=V(net)$degree, layout=layout.fruchterman.reingold,
     edge.width = E(net)$weight/3)

##############################
#Calcular subgrupos
##############################

###############################
# Cliques
##############################

#En este caso no es necesario, pero habría que convertir el grafo en no dirigido para hacer estos
#cálculos porque Igraph trata la red como no dirigida cuando hace las operaciones
#de cliques
#net.sym <- as.undirected(net, mode="collapse",edge.attr.comb = list(weight="sum","ignore"))
net.sym <- net

#Encontrar el número de cliques igual o mayores de 3 usando el método  count_max_cliques()
count_max_cliques(net.sym, min=3)   

#Obtener el número de elementos del clique de mayor tamaño
clique_num(net.sym)                          

#Listar los cliques de tamaño 3 o superior
max_cliques(net.sym, min=3)                        

#También se puede definir una variable que los almacende y recorrerla
mc <- max_cliques(net.sym, min=3)
for(i in 1:length(mc)){               
  print(mc[[i]])
  plot(induced_subgraph(net.sym, mc[[i]]))
}

#Nos centramos en los cliques de mayor tamaño y lo almacenamos en un variable para recorrerlos
cliq <- largest_cliques(net.sym)             
for(i in 1:length(cliq)){               
  print(cliq[[i]])
}

#Fijar la pantalla en una fila y dos columnas y mostramos los cliques
par(mfrow = c(1, 2))                   
for(i in 1:length(cliq)){              
  plot(induced_subgraph(net.sym, cliq[[i]]))
}

#Volvemos a poner la pantalla con una columna
par(mfrow = c(1, 1))                 


###############################
# k-cores 
##############################

#Encontrar los k-cores, nodos que están conectados a otros k nodos
kcore <- coreness(net.sym)

#Mostrar los cores. Los nombres de los vértices está arriba y el core al que pertenece abajo
print(kcore)

#Añadir la propiedad kcore a los nodos de la red
V(net.sym)$kcore <- kcore     

print(V(net.sym)$kcore)

#Mostrar el gráfico donde el color de los nodos está en función del
#kcore
plot(net.sym, edge.arrow.size=.2, edge.arrow.width=0.6, edge.curved=0,
     vertex.color=V(net.sym)$kcore, vertex.frame.color="#555555",
     vertex.label=V(net)$label, vertex.label.color="black",
     vertex.label.cex=.5, vertex.size=V(net)$degree, layout=layout.fruchterman.reingold,
     edge.width = E(net.sym)$weight/3)


#Obtener uno o más subgrafos basados en k-cores

#Mostrar los nodos incluidos en cada k-core
table(kcore)

#Mostrar el 4-core usando la función induced_subgraph().
g4c <- induced_subgraph(net.sym, kcore==3)


##Mostrar la gráfica
plot(g4c)
 

###############################
# bloque Cohesionados 
##############################

#Encontrar bloques cohesionados, comienza con componentes e identifica
#grandes subestructuras dentro de los componentes, luego en las grandes
#estructuras identificar subestructuras más pequeñas hasta alcanzar a los cliques
blocks <- cohesive.blocks(net.sym, labels=T)

#Los bloques encontrados y los nodos que están incluidos
subblocks <- blocks(blocks)
print(subblocks)

#Informa del score de cohesión para cada bloque. Indica el número de nodos
#que debes borrar para que no sea conectado
cohesion(blocks) 

#Mostrar la estructura jerarquica de los bloques
plotHierarchy(blocks) 

for(i in 1:length(subblocks)){               
  plot(induced_subgraph(net.sym, subblocks[[i]]))
}


###############################
# Detección de grupos
##############################
  
#Agrupar los que tengan un mayor número de enlaces entre ellos
communityMulti <- multilevel.community(net.sym)

#Número de grupos que ha encontrado
length(communityMulti)

#Imprimir los grupos
print(communityMulti)

#Mostrar en función del grupo (sombreado)
plot(net.sym, vertex.size = 3, vertex.label = NA, mark.groups = communityMulti)

#Mostrar en función del grupo (color del nodo)
V(net.sym)$color <- membership(communityMulti)
plot(net.sym, vertex.size = 3, vertex.label = NA)

