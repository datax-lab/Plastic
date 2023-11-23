from tools.encoding_cnn_renew import Encoding

class Config:
    epoch = 30

    batch_size_list = [16, 32, 64]
    lr_list = [1e-4, 1e-5, 1e-6]
    
    encoding = Encoding()
    
    method_list = [encoding.zScales, encoding.VHSE, encoding.PCscores, encoding.sScales, encoding.one_hot_encoding, encoding.five_bit_encoding, 
                   encoding.six_bit_encoding, encoding.hydrohobicity_matrix, encoding.meiler_parameters, 
                   encoding.acthely_factors, encoding.pam250, encoding.blosum62, encoding.cristian, 
                   encoding.tanaka_sheraga, encoding.miyazawa_energies, encoding.micheletti_potentials, 
                   encoding.aesnn3, encoding.ann4d]
    
    rm_method_list = [encoding.zScales, encoding.VHSE, encoding.PCscores, encoding.sScales, encoding.one_hot_encoding, encoding.five_bit_encoding, 
                   encoding.six_bit_encoding, encoding.hydrohobicity_matrix, encoding.meiler_parameters, 
                   encoding.acthely_factors, encoding.pam250, encoding.blosum62, encoding.cristian, 
                   encoding.tanaka_sheraga, encoding.miyazawa_energies, encoding.micheletti_potentials, 
                   encoding.aesnn3, encoding.ann4d]
    
    method_name_list = ['zScales', 'VHSE', 'PCscores', 'sScales', 'one_hot_encoding', 'five_bit_encoding', 
                        'six_bit_encoding', 'hydrohobicity_matrix', 'meiler_parameters', 
                        'acthely_factors', 'pam250', 'blosum62', 'cristian', 'tanaka_sheraga', 
                        'miyazawa_energies', 'micheletti_potentials', 'aesnn3', 'ann4d']
    kr_size_list = [5, 8, 11, 11, 20, 5, 6, 20, 7, 5, 20, 20, 20, 20, 20, 20, 3, 4]

opt = Config()
    
