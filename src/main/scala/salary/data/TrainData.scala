package salary.data

import com.mongodb.casbah.Imports._

object TrainData extends BaseParser {

  def createRecord(header: Array[String], record: Array[String]) = {
    if (record.length > 12)
      throw new IllegalArgumentException("The record must have at most 12 fields")

    val obj = MongoDBObject(
      header(0) -> record(0).toLong,
      header(10) -> record(10).toInt
    )
    for (i <- List.range(0, 12))
      if (i != 0 && i != 10)
        obj += header(i) -> record(i)

    obj
  }

}
