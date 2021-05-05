classDiagram
    main--|> file_details
    file_details --|> cell_data
    cell_data --|> protocols
    protocols --|> abf_file
    abf_file --|> Analysed__info
    abf_file --|> Save_result
    Analysed__info --|> Save_result
    cell_data :list_files(on file path)
    cell_data : +List cell_no
    cell_data : +List cell_data
    cell_data : +String gender
    cell_data: +isMammal()
    cell_data: +mate()
    class main{
      +String beakColor
      +swim()
      +quack()
    }
    class file_details{
      +String beakColor
      +swim()
      +quack()
    }
    class cell_data{
      -int sizeInFeet
      -canEat()
    }
    class protocols{
      +bool is_wild
      +run()
    }
    class abf_file{
      +bool is_wild
      +run()
    }
    class Analysed__info{
      +bool is_wild
      +run()
    }
    class Save_result{
      +bool is_wild
      +run()
    }   
