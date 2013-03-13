package salary.data

import java.io.FileReader

import au.com.bytecode.opencsv.CSVReader
import com.mongodb.casbah.Imports._

trait BaseParser {

  def createRecord(header: Array[String], record: Array[String]): MongoDBObject

  def parseFile(filename: String, coll: MongoCollection) = {
    val reader = new CSVReader(new FileReader(filename))
    try {
      var isFirst = true
      var header: Array[String] = null
      var line: Array[String] = reader.readNext
      var index: Int = 0
      while (line != null) {
        if (isFirst) {
          header = line
          isFirst = false
        } else {
          coll += createRecord(header, line)
          if (index % 10000 == 0)
            println("%s: %d".format(filename, index))
        }
        line = reader.readNext
        index += 1
      }
      for (elem <- header)
        println(elem)
    } finally {
      reader.close
    }
  }

}
