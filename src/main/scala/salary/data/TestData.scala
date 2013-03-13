package salary.data

import com.mongodb.casbah.Imports._

object TestData extends BaseParser {

  def createRecord(header: Array[String], record: Array[String]) = {
    if (record.length > 10)
      throw new IllegalArgumentException("The record must have at most 10 fields")

    val obj = MongoDBObject(
      header(0) -> record(0).toLong
    )
    for (i <- List.range(0, 10))
      if (i != 0)
        obj += header(i) -> record(i)

    obj
  }

}
