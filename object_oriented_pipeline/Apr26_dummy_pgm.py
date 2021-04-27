
# keep in the order used fo rhte analysis
protocol_list = ["bl", "p1", "tr"...]


known_protocols = { "bl":"baseline.pro", "tr":"training.pro", "p1":"proto1", "proto2" }




def fn1( proto_obj ):
    # stuff here

def fn2( proto_obj ):
    # other stuff here

known_functions = { "bl": fn1, "p1": fn2 }

class Protocol:
    def __init__( self, name, pro, abf, func, pattern = "" ):
        self.name = name
        self.pro = pro
        self.abf = abf

        self.stim_trace = readProtocol( pro )
        self.data = readABF( abf )
        #self.func = lookup_func( name )
        self.func = known_functions[name]
        self.pattern = lookup_pattern( name )

        def analyze(self):
            self.func( args )


def main():

    # Reading and building the protocols
    protocols = {}
    for abf in os.listdir( dirname ):
        if abf.suffix == 'abf':
            fd = open( abf, "r" ):
                proto_filename_name = get_protocol_from_file( fd )
                proto_name = lookup( known_protocols, proto_filename )

                protocols[proto_name] = Protocol( proto_name, proto_file_name, abf ) 


    # First pass analysis

    for pr in protocol_list:
        protocols[pr].analyze()


    # second pass analysis
    # Might involve newly generated data structures.


